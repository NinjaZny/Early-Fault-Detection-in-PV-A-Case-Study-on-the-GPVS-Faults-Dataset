import os
import itertools
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
 
# ==========================
# Global settings
# ==========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 200              # Max number of epochs during each training run
EARLY_STOPPING_PATIENCE = 15  # Stop training if validation loss doesn't improve for this many epochs
K_FOLDS = 3                   # K-fold cross validation

# Hyperparameter grid.
# Note: invalid structures (enc1_dim < enc2_dim or enc2_dim < latent_dim)
# will be filtered out later.
HYPERPARAM_GRID = {
    "enc1_dim":   [32, 64, 128],
    "enc2_dim":   [16, 32, 64],
    "latent_dim": [8, 16, 32],
    "learning_rate": [1e-3, 5e-4],
    "batch_size": [64]
}


# ==========================
# Model definition
# ==========================
# A straightforward LSTM autoencoder:
# - Two encoder layers, two decoder layers
# - Hidden size shrinks in the encoder and expands in the decoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=15, seq_len=200,
                 enc1_dim=128, enc2_dim=64, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_dim, enc1_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(enc1_dim, enc2_dim, batch_first=True)
        self.to_latent = nn.Linear(enc2_dim, latent_dim)

        self.seq_len = seq_len

        # Decoder: latent vector → enc2 → enc1 → original input dim
        self.latent_to_dec = nn.Linear(latent_dim, enc2_dim)
        self.decoder_lstm1 = nn.LSTM(enc2_dim, enc1_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(enc1_dim, input_dim, batch_first=True)

    def forward(self, x):
        # x has shape [batch, seq_len, input_dim]

        # ---- Encoder ----
        out1, _ = self.encoder_lstm1(x)
        _, (h_n, _) = self.encoder_lstm2(out1)
        z = self.to_latent(h_n[-1])   # latent representation

        # ---- Decoder ----
        dec_init = self.latent_to_dec(z)
        dec_input = dec_init.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out1, _ = self.decoder_lstm1(dec_input)
        dec_out2, _ = self.decoder_lstm2(dec_out1)

        return dec_out2


# ==========================
# Helper functions
# ==========================
def _ensure_timeseries_shape(X: np.ndarray) -> np.ndarray:
    """
    Ensure the data is in [N, T, D] format.
    If the second and third axes look swapped (D > T), flip them.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected 3D array [N, T, D], got {X.shape}")

    # In most cases T (e.g., 200) is larger than D (~15).
    # If that doesn’t hold, assume the array is [N, D, T] and transpose it.
    if X.shape[1] < X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
    return X


def _train_one_run(X_train_tensor: torch.Tensor,
                   train_indices: np.ndarray,
                   val_indices: np.ndarray,
                   enc1_dim: int,
                   enc2_dim: int,
                   latent_dim: int,
                   learning_rate: float,
                   batch_size: int,
                   max_epochs: int,
                   patience: int) -> float:
    """
    Train a model for a single train/validation split and return the best
    validation loss achieved during this run.
    Used internally by the K-fold cross validation loop.
    """
    loss_fn = nn.MSELoss()

    X_train_split = X_train_tensor[train_indices]
    X_val_split   = X_train_tensor[val_indices]

    train_loader = DataLoader(TensorDataset(X_train_split, X_train_split),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_split, X_val_split),
                              batch_size=batch_size)

    seq_len   = X_train_split.shape[1]
    input_dim = X_train_split.shape[2]

    model = LSTMAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        enc1_dim=enc1_dim,
        enc2_dim=enc2_dim,
        latent_dim=latent_dim
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                output = model(batch_x)
                loss = loss_fn(output, batch_x)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses))

        # Early stopping
        if mean_val_loss < best_val_loss - 1e-6:
            best_val_loss = mean_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss


def _iter_hyperparam_configs(grid: dict):
    """
    Generate every hyperparameter combination from the grid,
    but skip structures that don’t respect the bottleneck constraint:

        enc1_dim >= enc2_dim >= latent_dim

    This avoids building architectures that expand inward.
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))

        if not (config["enc1_dim"] >= config["enc2_dim"] >= config["latent_dim"]):
            continue

        yield config


# ==========================
# Main CV + final training
# ==========================
def train_lstm_ae_with_cv(pipeline: str,
                          data_dir: str = "data/processed/",
                          artifacts_dir: str = "artifacts/"):
    """
    Full workflow:
      1. Load the healthy LPPT data for the given pipeline.
      2. Run K-fold cross validation for each hyperparameter combination.
      3. Pick the configuration with the lowest average validation loss.
      4. Retrain a final model on the full dataset using the chosen hyperparameters.
      5. Save the model and logs.
    """
    # ---- Load training data ----
    x_train_path = os.path.join(data_dir, f"X_train_LPPT_{pipeline}.npy")
    if not os.path.exists(x_train_path):
        raise FileNotFoundError(f"Training file not found: {x_train_path}")

    X_train = np.load(x_train_path)
    X_train = _ensure_timeseries_shape(X_train)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    n_samples = X_train_tensor.shape[0]
    print(f"[CV] Loaded {n_samples} healthy LPPT windows for hyperparameter search")

    # ---- K-fold CV ----
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    cv_results = []

    for config in _iter_hyperparam_configs(HYPERPARAM_GRID):
        print(f"\n[CV] Testing config: {config}")

        fold_losses = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            print(f"[CV]   Fold {fold_idx+1}/{K_FOLDS}")

            fold_val_loss = _train_one_run(
                X_train_tensor=X_train_tensor,
                train_indices=train_idx,
                val_indices=val_idx,
                enc1_dim=config["enc1_dim"],
                enc2_dim=config["enc2_dim"],
                latent_dim=config["latent_dim"],
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                max_epochs=MAX_EPOCHS,
                patience=EARLY_STOPPING_PATIENCE
            )

            print(f"[CV]   Fold {fold_idx+1} best val_loss = {fold_val_loss:.6f}")
            fold_losses.append(fold_val_loss)

        mean_cv_loss = float(np.mean(fold_losses))
        print(f"[CV] Config {config} → mean val_loss = {mean_cv_loss:.6f}")

        cv_results.append((config, mean_cv_loss))

    # ---- Pick best config ----
    if not cv_results:
        raise RuntimeError("No valid hyperparameter configurations passed the dimensionality checks.")

    cv_results.sort(key=lambda x: x[1])
    best_config, best_loss = cv_results[0]

    print(f"\n[CV] Best configuration: {best_config} (mean val_loss = {best_loss:.6f})")

    os.makedirs(os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}"), exist_ok=True)

    # Save CV results
    cv_path = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "cv_results.csv")
    pd.DataFrame([
        dict(
            enc1_dim=c["enc1_dim"],
            enc2_dim=c["enc2_dim"],
            latent_dim=c["latent_dim"],
            learning_rate=c["learning_rate"],
            batch_size=c["batch_size"],
            mean_val_loss=loss
        )
        for c, loss in cv_results
    ]).to_csv(cv_path, index=False)

    # ---- Final training ----
    print("\n[Final Train] Training on full dataset with best hyperparameters...")

    full_loader = DataLoader(
        TensorDataset(X_train_tensor, X_train_tensor),
        batch_size=best_config["batch_size"],
        shuffle=True
    )

    seq_len   = X_train_tensor.shape[1]
    input_dim = X_train_tensor.shape[2]

    model = LSTMAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        enc1_dim=best_config["enc1_dim"],
        enc2_dim=best_config["enc2_dim"],
        latent_dim=best_config["latent_dim"]
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=best_config["learning_rate"])
    loss_fn = nn.MSELoss()

    best_train_loss = float("inf")
    patience_counter = 0

    log_path = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "training_log_final.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

        for batch_x, _ in tqdm(full_loader, desc=f"[Final Train] Epoch {epoch}"):
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss}\n")

        print(f"[Final Train] Epoch {epoch}: Train Loss = {train_loss:.6f}")

        if train_loss < best_train_loss - 1e-6:
            best_train_loss = train_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "model_best_cv.ckpt")
            )
            print("[Final Train] ---- Updated best model ----")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("[Final Train] Early stopping triggered.")
                break

    # ---- Plot final training curve ----
    log_df = pd.read_csv(log_path)
    plt.figure(figsize=(8, 5))
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Final Training Curve (Best CV Config) - {pipeline}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "loss_curve_final.png"), dpi=200)
    plt.close()
