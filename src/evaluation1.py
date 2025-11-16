import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_autoencoder import LSTMAutoencoder, _ensure_timeseries_shape

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/processed"
ARTIFACTS_DIR = "artifacts"

BATCH_SIZE = 256
THRESHOLD_QUANTILE = 0.95


def _get_model_dir(pipeline: str) -> str:
    """Return artifact directory for a given pipeline."""
    return os.path.join(ARTIFACTS_DIR, f"LSTM-AE__{pipeline}")


def _load_best_config(pipeline: str):
    """Load best hyperparameter config from CV results."""
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
    """Build model with best config and load trained weights."""
    # infer seq_len and input_dim from training data
    x_train_path = os.path.join(DATA_DIR, f"X_train_LPPT_{pipeline}.npy")
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

    ckpt = os.path.join(_get_model_dir(pipeline), "model_best_cv.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def _recon_errors(model, X):
    """Compute per-sequence reconstruction MSE errors."""
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
    """Evaluate normal vs fault detection for a given pipeline."""
    print("=== EVALUATION: Normal vs Abnormal Detection ===")

    # ---- load model and compute threshold from validation set ----
    config = _load_best_config(pipeline)
    model = _build_model_from_config(config, pipeline)
    model_dir = _get_model_dir(pipeline)
    os.makedirs(model_dir, exist_ok=True)

    X_val = np.load(os.path.join(DATA_DIR, f"X_val_LPPT_{pipeline}.npy"))
    val_errs = _recon_errors(model, X_val)
    threshold = float(np.quantile(val_errs, THRESHOLD_QUANTILE))
    print(f"[Eval] Threshold = {threshold:.6e}")

    # ---- normal test data ----
    X_test_normal = np.load(os.path.join(DATA_DIR, f"X_test_LPPT_{pipeline}.npy"))
    normal_errs = _recon_errors(model, X_test_normal)
    preds_normal = (normal_errs >= threshold).astype(int)

    fp = int((preds_normal == 1).sum())
    tn = int((preds_normal == 0).sum())
    fpr = fp / (fp + tn)
    print(f"[Normal] FP={fp}, TN={tn}, FPR={fpr:.4f}")

    all_scores = [normal_errs]
    all_labels = [np.zeros_like(normal_errs, dtype=int)]

    rows_scores = [
        dict(split="test", fault="Normal", err=float(e), label=0)
        for e in normal_errs
    ]

    # ---- fault test data ----
    fault_stats = []

    for fname in sorted(os.listdir(DATA_DIR)):
        if not (fname.startswith("X_test_F") and "_LPPT_" in fname and fname.endswith(".npy")):
            continue

        flabel = fname.split("_")[2]  # e.g. F1L / F2L
        X_fault = np.load(os.path.join(DATA_DIR, fname))
        errs_fault = _recon_errors(model, X_fault)
        preds_fault = (errs_fault >= threshold).astype(int)

        tp = int((preds_fault == 1).sum())
        fn = int((preds_fault == 0).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"[Fault {flabel}] TP={tp}, FN={fn}, TPR={tpr:.4f}")

        fault_stats.append(dict(fault=flabel, TP=tp, FN=fn, TPR=tpr))
        all_scores.append(errs_fault)
        all_labels.append(np.ones_like(errs_fault, dtype=int))

        for e in errs_fault:
            rows_scores.append(
                dict(split="test", fault=flabel, err=float(e), label=1)
            )

    df_fault = pd.DataFrame(fault_stats)
    df_fault.to_csv(os.path.join(model_dir, "fault_detection_stats.csv"), index=False)

    # ---- global ROC / PR metrics ----
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    roc_auc = roc_auc_score(all_labels, all_scores)
    pr_auc = average_precision_score(all_labels, all_scores)
    print(f"[GLOBAL] ROC-AUC = {roc_auc:.4f}, PR-AUC = {pr_auc:.4f}")

    # ROC curve
    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(fpr_arr, tpr_arr, label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # PR curve
    prec_arr, rec_arr, _ = precision_recall_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(rec_arr, prec_arr, label=f"PR (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "pr_curve.png"), dpi=300)
    plt.close()

    # ---- error distribution (normal vs fault) ----
    df_scores = pd.DataFrame(rows_scores)
    df_scores.to_csv(os.path.join(model_dir, "all_test_scores.csv"), index=False)

    normal_vals = df_scores[df_scores["label"] == 0]["err"].values
    fault_vals = df_scores[df_scores["label"] == 1]["err"].values

    plt.figure()
    plt.hist(normal_vals, bins=50, alpha=0.6, density=True, label="Normal")
    plt.hist(fault_vals, bins=50, alpha=0.6, density=True, label="Fault")
    plt.yscale("log")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density (log scale)")
    plt.title("Error Distribution: Normal vs Fault")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "error_hist.png"), dpi=300)
    plt.close()

    # ---- global precision / recall at chosen threshold ----
    TP_global = sum(item["TP"] for item in fault_stats)
    FN_global = sum(item["FN"] for item in fault_stats)
    FP_global = fp  # FP comes only from normal windows

    precision_global = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else 0.0
    recall_global = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else 0.0

    print(f"[GLOBAL] Precision = {precision_global:.6f}")
    print(f"[GLOBAL] Recall    = {recall_global:.6f}")
    print("=== Evaluation Finished ===")
