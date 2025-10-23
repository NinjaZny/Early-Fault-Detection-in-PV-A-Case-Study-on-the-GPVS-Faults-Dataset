import os
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Global training settings
BATCH_SIZE = 64
EPOCHS = 400
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A standard LSTM Autoencoder:
# - 2 encoder layers + 2 decoder layers
# - symmetric structure with decreasing and then increasing hidden sizes
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=15, seq_len=200, enc1_dim=128, enc2_dim=64, latent_dim=32):
        super().__init__()
        
        # Encoder part
        self.encoder_lstm1 = nn.LSTM(input_dim, enc1_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(enc1_dim, enc2_dim, batch_first=True)
        self.to_latent = nn.Linear(enc2_dim, latent_dim)  # convert hidden state to a latent vector

        self.seq_len = seq_len

        # Decoder part: latent -> enc2 -> enc1 -> input_dim
        self.latent_to_dec = nn.Linear(latent_dim, enc2_dim)
        self.decoder_lstm1 = nn.LSTM(enc2_dim, enc1_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(enc1_dim, input_dim, batch_first=True)
        # output of the last LSTM directly matches the original feature dimensions

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # ---- Encoder ----
        out1, _ = self.encoder_lstm1(x)                        # [batch, seq_len, 64]
        _, (h_n, _) = self.encoder_lstm2(out1)                 # h_n: [1, batch, 32]
        z = self.to_latent(h_n[-1])                            # get latent vector [batch, 16]

        # ---- Decoder ----
        dec_init = self.latent_to_dec(z)                       # map latent to decoder init [batch, 32]
        # repeat across time steps (one same vector per timestamp)
        dec_input = dec_init.unsqueeze(1).repeat(1, self.seq_len, 1)  
        dec_out1, _ = self.decoder_lstm1(dec_input)            # [batch, seq_len, 64]
        dec_out2, _ = self.decoder_lstm2(dec_out1)             # [batch, seq_len, 15]
        
        return dec_out2


def train_lstm_ae(pipeline, data_dir="data/processed/", artifacts_dir="artifacts/"):
    # Load datasets from .npy files
    x_train_path = os.path.join(data_dir, f"X_train_LPPT_{pipeline}.npy")
    x_val_path = os.path.join(data_dir, f"X_val_LPPT_{pipeline}.npy")
    X_train = np.load(x_train_path)
    X_val = np.load(x_val_path)

    # Convert arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # Create datasets (input = target since it’s an autoencoder)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    # DataLoaders for batch training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Build model
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # methods of learning rate scheduling to try: (may not be useful because of the size of model)
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-3, cooldown=3, min_lr=1e-5, verbose=True)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=5, # first restart epochs
    #     T_mult=2, # multiply T_i by T_mult after a restart
    #     eta_min=1e-5
    # )
    
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    # Make log directory and file
    os.makedirs(os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}"), exist_ok=True)
    log_path = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "training_log.csv")

    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        # Training loop
        for batch_x, _ in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()

            output = model(batch_x)
            loss = loss_fn(output, batch_x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                output = model(batch_x)
                loss = loss_fn(output, batch_x)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        # scheduler.step()
        # scheduler.step(val_losses[-1])
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)

        # Write to log
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch},{train_loss},{val_loss}\n")

        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "model.ckpt"))
            print("-----------------------Model checkpoint saved-----------------------")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break
    
    # After training loop
    log_df = pd.read_csv(log_path)

    plt.figure(figsize=(8,5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Curve - {pipeline}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "loss_curve.png"), dpi=200)
    plt.close()