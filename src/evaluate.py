import os
import argparse
import json
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.train_autoencoder import LSTMAutoencoder



def _extract_state_dict_from_checkpoint(ckpt_path: str, map_location: str = 'cpu') -> Dict[str, torch.Tensor]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)
    sd = None
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                sd = state[key]
                break
        if sd is None:
            sd = state
    else:
        raise RuntimeError("Unsupported checkpoint format.")
    return {k.replace("module.", ""): v for k, v in sd.items()}


def _infer_model_params_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    k_enc1 = "encoder_lstm1.weight_ih_l0"
    if k_enc1 not in sd:
        candidates = [k for k in sd.keys() if "encoder_lstm1" in k and "weight_ih" in k]
        if not candidates:
            raise KeyError(f"Expected key like '{k_enc1}'. Found: {list(sd.keys())[:10]}")
        k_enc1 = candidates[0]
    s = sd[k_enc1].shape
    enc2_key = [k for k in sd.keys() if 'encoder_lstm2.weight_ih' in k][0]
    latent_key = [k for k in sd.keys() if 'to_latent.weight' in k][0]
    return {
        "input_dim": int(s[1]),
        "enc1_dim": int(s[0] // 4),
        "enc2_dim": int(sd[enc2_key].shape[0] // 4),
        "latent_dim": int(sd[latent_key].shape[0]),
    }



#  build_model_from_checkpoint
#  Purpose: reconstruct model architecture from checkpoint and load weights
#  Inputs: ckpt_path, seq_len (for decoder repeat), device (my pc has some restriction)
#  Outputs: instantiated nn.Module with weights loaded (moved to device if provided)
#  Notes:
#    - Uses the inferred dims to instantiate LSTMAutoencoder exactly matching saved weights.

def build_model_from_checkpoint(ckpt_path: str, seq_len: int, device: torch.device = None) -> nn.Module:
    sd = _extract_state_dict_from_checkpoint(ckpt_path, 'cpu')
    params = _infer_model_params_from_state_dict(sd)

    model = LSTMAutoencoder(
        input_dim=params["input_dim"],
        seq_len=seq_len,
        enc1_dim=params["enc1_dim"],
        enc2_dim=params["enc2_dim"],
        latent_dim=params["latent_dim"]
    )
    model.load_state_dict(sd)
    if device:
        model.to(device)
    return model



#  evaluate_model_on_array
#  Purpose: run model inference on numpy array windows and return per-window MSEs
#  Inputs:
#    - model: nn.Module mapping (batch, seq, feat) -> (batch, seq, feat)
#    - X: numpy array shape (n, seq_len, feat)
#    - device: torch.device for inference
#    - batch_size: DataLoader batch size
#  Outputs: 1D numpy array of length n with mean-squared-error per window
#  Math/shape note:
#    - se = (out - xb)**2 with shape (batch, seq, feat)
#    - mse per sample = mean over time & features -> se.mean(dim=(1,2))

def evaluate_model_on_array(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.to(device)
    model.eval()
    if X.ndim != 3:
        raise ValueError(f"Expected [n, seq, feat], got {X.shape}")
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    errs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            se = (out - xb) ** 2
            mse = se.mean(dim=(1, 2)).cpu().numpy()
            errs.append(mse)
    return np.concatenate(errs) if errs else np.array([])



#  summarize_errors
#  Purpose: produce concise statistics from per-window MSEs
#  Inputs: errors (1D numpy)
#  Outputs: dict with count, mean, median, std, max and several percentiles
#  Notes:
#    - returns NaN-filled dict if input is empty for downstream robustness
def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    """Compute all required statistics from reconstruction MSE."""
    if errors.size == 0:
        return {k: np.nan for k in [
            "n_samples", "mean_mse", "median_mse", "std_mse", "max_mse",
            "p50_mse", "p75_mse", "p90_mse", "p95_mse", "p99_mse"
        ]}
    return {
        "n_samples": int(errors.size),
        "mean_mse": float(np.mean(errors)),
        "median_mse": float(np.median(errors)),
        "std_mse": float(np.std(errors)),
        "max_mse": float(np.max(errors)),
        "p50_mse": float(np.percentile(errors, 50)),
        "p75_mse": float(np.percentile(errors, 75)),
        "p90_mse": float(np.percentile(errors, 90)),
        "p95_mse": float(np.percentile(errors, 95)),
        "p99_mse": float(np.percentile(errors, 99)),
    }



#  safe_makedirs
#  Purpose: idempotent directory creation
#  Inputs: path string
#  Outputs: None (creates dir if missing)
def safe_makedirs(d):
    os.makedirs(d, exist_ok=True)



#  evaluate_pipeline
#  Purpose: end-to-end evaluation: infer shapes, load model, compute threshold from val,
#           evaluate all test files, save per-window and aggregate metrics.
#  Inputs:
#    - pipeline (str): name used to locate artifacts and data files
#    - device (torch.device|None)
#    - data_dir, artifacts_dir, results_dir (paths)
#    - threshold_percentile (float): e.g., 95.0
#    - batch_size (int)
#  Outputs: pandas.DataFrame with one row per evaluated test file and metrics
#
#  Short workflow / shape notes:
#    1) find a sample X_*_{pipeline}.npy to infer seq_len and feat_dim (expects 3D array)
#    2) build model from artifacts/LSTM-AE__{pipeline}/model.ckpt
#    3) compute val_errors by running evaluate_model_on_array on X_val files and set
#       threshold = percentile(val_errors, threshold_percentile)
#    4) for each X_test_*_{pipeline}.npy compute per-window MSEs, summary stats,
#       percent windows > threshold, and save per-sample CSV + aggregate CSV/XLSX.
def evaluate_pipeline(
    pipeline: str,
    device: torch.device = None,
    data_dir: str = "data/processed",
    artifacts_dir: str = "../artifacts",
    results_dir: str = "results",
    threshold_percentile: float = 95.0,
    batch_size: int = 256
) -> pd.DataFrame:

    model_dir = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}")
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    log_path = os.path.join(model_dir, "training_log.csv")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[evaluate] Pipeline: {pipeline}")
    print(f"   Data dir: {os.path.abspath(data_dir)}")
    print(f"   Artifacts: {os.path.abspath(artifacts_dir)}")

    # --- 1. FIND SAMPLE FILE TO INFER SHAPE ---
    sample_paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(f"_{pipeline}.npy") and f.startswith(("X_val", "X_test")):
                sample_paths.append(os.path.join(root, f))

    if not sample_paths:
        raise FileNotFoundError(f"No X_val/X_test files in {data_dir}")

    sample_path = sample_paths[0]
    X_sample = np.load(sample_path)
    if X_sample.ndim != 3:
        raise ValueError(f"Sample {sample_path} must be 3D")
    seq_len, feat_dim = X_sample.shape[1], X_sample.shape[2]
    print(f"   Inferred: seq_len={seq_len}, feat={feat_dim}")

    # --- 2. LOAD MODEL ---
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model not found: {ckpt_path}")
    model = build_model_from_checkpoint(ckpt_path, seq_len=seq_len, device=device)

    # --- 3. FIND TEST FILES ---
    test_files = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.startswith("X_test_") and f.endswith(f"_{pipeline}.npy"):
                test_files.append(os.path.join(root, f))
    test_files = sorted(test_files)
    if not test_files:
        raise FileNotFoundError(f"No test files in {data_dir}")


    # NEW CODE: Compute MSE on actual validation data
    val_errors = np.array([])
    val_files = [p for p in sample_paths if "X_val" in p]
    if not val_files:
        print("   WARNING: No X_val files found. Using X_test_LPPT/MPPT as proxy.")
        val_files = [p for p in sample_paths if "X_test_LPPT" in p or "X_test_MPPT" in p][:1]

    print(f"   Computing threshold from {len(val_files)} validation file(s)...")
    for vf in val_files:
        Xv = np.load(vf)
        errs = evaluate_model_on_array(model, Xv, device, batch_size)
        val_errors = np.concatenate([val_errors, errs]) if val_errors.size else errs

    if val_errors.size == 0:
        raise ValueError("Could not compute validation errors.")

    threshold = np.percentile(val_errors, threshold_percentile)
    print(f"   Threshold p{int(threshold_percentile)} = {threshold:.6f} (from {val_errors.size} samples)")

    # --- 4. EVALUATE EACH TEST FILE ---
    rows = []
    safe_makedirs(results_dir)

    for tf in test_files:
        name = os.path.basename(tf).replace("X_test_", "").replace(f"_{pipeline}.npy", "")
        print(f"   → Evaluating: {name}")

        X = np.load(tf)
        if X.ndim != 3:
            print(f"   Skipping {name}: invalid shape {X.shape}")
            continue

        errors = evaluate_model_on_array(model, X, device, batch_size)
        stats = summarize_errors(errors)


        stats.update({
            "pipeline": pipeline,
            "test_name": name,
            f"threshold_p{int(threshold_percentile)}": threshold,
            "pct_windows_over_threshold": 100 * np.mean(errors > threshold)
        })

        rows.append(stats)

        # Save per-window errors
        pd.DataFrame({
            "mse": errors,
            "is_anomaly": errors > threshold
        }).to_csv(
            os.path.join(results_dir, f"per_sample_errors_LSTM-AE__{pipeline}_{name}.csv"),
            index=False
        )

    # --- 5. SAVE RESULTS ---
    df = pd.DataFrame(rows)

    # Preferred column order (without lead_time)
    preferred_cols = [
        "pipeline", "test_name",
        "n_samples", "mean_mse", "median_mse", "std_mse",
        "p50_mse", "p75_mse", "p90_mse", "p95_mse", "p99_mse", "max_mse",
        f"threshold_p{int(threshold_percentile)}",
        "pct_windows_over_threshold"
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    df = df[cols]

    # Save CSV
    metrics_path = os.path.join(results_dir, f"metrics_LSTM-AE__{pipeline}.csv")
    df.to_csv(metrics_path, index=False)
    print(f"   Saved: {metrics_path}")

    # Update global XLSX
    xlsx_path = os.path.join(results_dir, "results_summary.xlsx")
    if os.path.exists(xlsx_path):
        try:
            old = pd.read_excel(xlsx_path)
            df = pd.concat([old, df], ignore_index=True)
        except:
            pass
    df = df.drop_duplicates(["pipeline", "test_name"], keep="last")
    df.to_excel(xlsx_path, index=False)

    # Save metadata
    json.dump({
        "pipeline": pipeline,
        "seq_len": seq_len,
        "feat_dim": feat_dim,
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "validation_samples_used": val_errors.size,
        "test_files": test_files
    }, open(os.path.join(results_dir, f"metadata_LSTM-AE__{pipeline}.json"), "w"), indent=2)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--artifacts_dir", type=str, default="../artifacts")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    args = parser.parse_args()

    evaluate_pipeline(
        pipeline=args.pipeline,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        results_dir=args.results_dir,
        threshold_percentile=args.threshold_percentile
    )
