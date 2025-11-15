# src/evaluation1.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_autoencoder import LSTMAutoencoder, _ensure_timeseries_shape

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"

BATCH_SIZE = 256
THRESHOLD_QUANTILE = 0.95


def _get_model_dir(pipeline: str) -> str:
    """
    Build the artifact directory path for a given pipeline.
    """
    return os.path.join(ARTIFACTS_DIR, f"LSTM-AE__{pipeline}")


def _load_best_config(pipeline: str):
    """
    Load the cross-validation results and retrieve the best config
    (the one with the minimum validation loss).
    """
    model_dir = _get_model_dir(pipeline)
    cv_path = os.path.join(model_dir, "cv_results.csv")
    cv_df = pd.read_csv(cv_path)
    row = cv_df.loc[cv_df["mean_val_loss"].idxmin()]

    return dict(
        enc1_dim=int(row.enc1_dim),
        enc2_dim=int(row.enc2_dim),
        latent_dim=int(row.latent_dim),
        learning_rate=float(row.learning_rate),
        batch_size=int(row.batch_size),
    )


def _build_model_from_config(config, pipeline: str):
    """
    Rebuild the model using the best architecture configuration
    and load its trained weights.
    """
    # Load training data to infer seq_len and input_dim.
    x_train_path = os.path.join(DATA_DIR, f"X_train_LPPT_{pipeline}.npy")
    X_train = np.load(x_train_path)
    X_train = _ensure_timeseries_shape(X_train)

    seq_len = X_train.shape[1]
    input_dim = X_train.shape[2]

    # Initialize model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        seq_len=seq_len,
        enc1_dim=config["enc1_dim"],
        enc2_dim=config["enc2_dim"],
        latent_dim=config["latent_dim"],
    ).to(DEVICE)

    # Load best checkpoint
    model_dir = _get_model_dir(pipeline)
    ckpt = os.path.join(model_dir, "model_best_cv.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    return model


@torch.no_grad()
def _recon_errors(model, X):
    """
    Compute reconstruction errors (MSE per sequence) for a dataset.
    """
    X = _ensure_timeseries_shape(X)
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=BATCH_SIZE
    )

    errs = []
    for (batch,) in loader:
        batch = batch.to(DEVICE)
        out = model(batch)
        mse = torch.mean((batch - out) ** 2, dim=(1, 2))
        errs.append(mse.cpu().numpy())

    return np.concatenate(errs, axis=0)


def evaluate_binary_detection(pipeline: str):
    """
    Evaluate the model using a given pipeline name.
    The evaluation includes:
        - determining threshold from validation data
        - computing FP/TN on normal test data
        - computing TP/FN/TPR for each fault type (F1L–F7L)
    """
    print("=== EVALUATION: Normal vs Abnormal Detection ===")

    # 1. Load config and model
    config = _load_best_config(pipeline)
    model = _build_model_from_config(config, pipeline)

    # 2. Compute threshold using validation reconstruction errors
    X_val = np.load(os.path.join(DATA_DIR, f"X_val_LPPT_{pipeline}.npy"))
    val_errs = _recon_errors(model, X_val)
    threshold = float(np.quantile(val_errs, THRESHOLD_QUANTILE))
    print(f"[Eval] Threshold = {threshold:.6e}")

    # 3. Evaluate on normal test data
    X_test_normal = np.load(os.path.join(DATA_DIR, f"X_test_LPPT_{pipeline}.npy"))
    normal_errs = _recon_errors(model, X_test_normal)
    preds_normal = (normal_errs >= threshold).astype(int)

    fp = int((preds_normal == 1).sum())
    tn = int((preds_normal == 0).sum())
    fpr = fp / (fp + tn)
    print(f"[Normal] FP={fp}, TN={tn}, FPR={fpr:.4f}")

    # 4. Evaluate fault cases (F1L–F7L)
    model_dir = _get_model_dir(pipeline)
    fault_stats = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith("X_test_F") and "_LPPT_" in fname and fname.endswith(".npy"):
            flabel = fname.split("_")[2]  # Extract label: F1L, F2L, ...
            X_fault = np.load(os.path.join(DATA_DIR, fname))

            errs_fault = _recon_errors(model, X_fault)
            preds_fault = (errs_fault >= threshold).astype(int)

            tp = int((preds_fault == 1).sum())
            fn = int((preds_fault == 0).sum())
            tpr = tp / (tp + fn)
            print(f"[Fault {flabel}] TP={tp}, FN={fn}, TPR={tpr:.4f}")

            fault_stats.append(dict(fault=flabel, TP=tp, FN=fn, TPR=tpr))

    # Store results
    pd.DataFrame(fault_stats).to_csv(
        os.path.join(model_dir, "fault_detection_stats.csv"),
        index=False
    )

    print("=== Evaluation Finished ===")
