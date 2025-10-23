import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 全局训练参数
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 主流LSTM-AE结构：2层编码, 2层解码, 对称递减/递增
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=15, seq_len=200, enc1_dim=64, enc2_dim=32, latent_dim=16):
        super().__init__()
        # 编码器部分
        self.encoder_lstm1 = nn.LSTM(input_dim, enc1_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(enc1_dim, enc2_dim, batch_first=True)
        self.to_latent = nn.Linear(enc2_dim, latent_dim)  # encoder输出最后hidden降维为潜变量

        self.seq_len = seq_len

        # 解码器部分（潜变量先升回32，再升回64）
        self.latent_to_dec = nn.Linear(latent_dim, enc2_dim)
        self.decoder_lstm1 = nn.LSTM(enc2_dim, enc1_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(enc1_dim, input_dim, batch_first=True)
        # 最后一层lstm输出直接就是原始特征的维度

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        # ---- Encoder ----
        out1, _ = self.encoder_lstm1(x)                              # [batch, seq_len, 64]
        _, (h_n, _) = self.encoder_lstm2(out1)                       # h_n: [1, batch, 32]
        z = self.to_latent(h_n[-1])                                  # [batch, 16]
        # ---- Decoder ----
        dec_init = self.latent_to_dec(z)                             # [batch, 32]
        dec_input = dec_init.unsqueeze(1).repeat(1, self.seq_len, 1) # [batch, seq_len, 32]
        dec_out1, _ = self.decoder_lstm1(dec_input)                  # [batch, seq_len, 64]
        dec_out2, _ = self.decoder_lstm2(dec_out1)                   # [batch, seq_len, 15]
        return dec_out2

def train_lstm_ae(pipeline, data_dir="data/", artifacts_dir="artifacts/"):
    # 加载数据集
    x_train_path = os.path.join(data_dir, f"X_train_LPPT_{pipeline}.npy")
    x_val_path = os.path.join(data_dir, f"X_val_LPPT_{pipeline}.npy")
    X_train = np.load(x_train_path)
    X_val = np.load(x_val_path)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]
    model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}"), exist_ok=True)
    log_path = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "training_log.csv")
    with open(log_path, "w") as log_file:
        log_file.write("epoch,train_loss,val_loss\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for batch_x, _ in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_x)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                output = model(batch_x)
                loss = loss_fn(output, batch_x)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        # 写日志
        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch},{train_loss},{val_loss}\n")

        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "model.ckpt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("[translate:提前停止] triggered")
                break

if __name__ == "__main__":
    train_lstm_ae("mad_z_win200_s100")
