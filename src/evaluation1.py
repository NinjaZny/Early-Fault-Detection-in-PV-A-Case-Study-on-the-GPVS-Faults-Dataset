# src/evaluation1.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_autoencoder import LSTMAutoencoder, _ensure_timeseries_shape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIPELINE = "IQR_minmax_butterworth_none_Slidingwindow_"

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, f"LSTM-AE__{PIPELINE}")

BATCH_SIZE = 256
THRESHOLD_QUANTILE = 0.95


def _load_best_config():
    cv_path = os.path.join(MODEL_DIR, "cv_results.csv")
    cv_df = pd.read_csv(cv_path)
    row = cv_df.loc[cv_df["mean_val_loss"].idxmin()]
    return dict(
        enc1_dim=int(row.enc1_dim),
        enc2_dim=int(row.enc2_dim),
        latent_dim=int(row.latent_dim),
        learning_rate=float(row.learning_rate),
        batch_size=int(row.batch_size),
    )


def _build_model_from_config(config):
    x_train_path = os.path.join(DATA_DIR, f"X_train_LPPT_{PIPELINE}.npy")
    X_train = np.load(x_train_path)
    X_train = _ensure_timeseries_shape(X_train)
    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]

    model = LSTMAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        enc1_dim=config["enc1_dim"],
        enc2_dim=config["enc2_dim"],
        latent_dim=config["latent_dim"],
    ).to(DEVICE)

    ckpt = os.path.join(MODEL_DIR, "model_best_cv.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def _recon_errors(model, X):
    X = _ensure_timeseries_shape(X)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=BATCH_SIZE)
    errs = []
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        mse = torch.mean((batch - out) ** 2, dim=(1, 2))
        errs.append(mse.cpu().numpy())
    return np.concatenate(errs, axis=0)


def evaluate_binary_detection():
    print("=== EVALUATION: Normal vs Abnormal Detection ===")

    # 1. load config/model
    config = _load_best_config()
    model = _build_model_from_config(config)

    # 2. threshold from validation
    X_val = np.load(os.path.join(DATA_DIR, f"X_val_LPPT_{PIPELINE}.npy"))
    val_errs = _recon_errors(model, X_val)
    threshold = float(np.quantile(val_errs, THRESHOLD_QUANTILE))
    print(f"[Eval] Threshold = {threshold:.6e}")

    # 3. evaluate normal test
    X_test_normal = np.load(os.path.join(DATA_DIR, f"X_test_LPPT_{PIPELINE}.npy"))
    normal_errs = _recon_errors(model, X_test_normal)
    preds_normal = (normal_errs >= threshold).astype(int)
    fp = int((preds_normal == 1).sum())
    tn = int((preds_normal == 0).sum())
    fpr = fp / (fp + tn)
    print(f"[Normal] FP={fp}, TN={tn}, FPR={fpr:.4f}")

    # 4. evaluate fault: F1L-F7L
    fault_stats = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("X_test_F") and "_LPPT_" in fname and fname.endswith(".npy"):
            flabel = fname.split("_")[2]  # F1L
            X_fault = np.load(os.path.join(DATA_DIR, fname))
            errs_fault = _recon_errors(model, X_fault)
            preds_fault = (errs_fault >= threshold).astype(int)

            tp = int((preds_fault == 1).sum())
            fn = int((preds_fault == 0).sum())
            tpr = tp / (tp + fn)
            print(f"[Fault {flabel}] TP={tp}, FN={fn}, TPR={tpr:.4f}")

            fault_stats.append(dict(fault=flabel, TP=tp, FN=fn, TPR=tpr))

    pd.DataFrame(fault_stats).to_csv(
        os.path.join(MODEL_DIR, "fault_detection_stats.csv"),
        index=False
    )

    print("=== Evaluation Finished ===")
