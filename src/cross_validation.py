import os
import pandas as pd
import torch

# Import modules
import preprocess
import train_autoencoder
import evaluate

# Device configuration for Apple M2
# PURPOSE: select MPS (Apple GPU) when available, otherwise CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === PARAMETER GRIDS ===
# enumerate preprocessing options to run in cross-validation
OUTLIER_METHODS = ['none', 'IQR']
NORMALIZE_METHODS = ['none', 'zscore', 'minmax', 'robust']
LOWPASS_FILTERS = ['none', 'butterworth', 'moving_average']
FEATURE_SELECTIONS = ['none', 'pca', 'robustpca']  # kernelpca removed: too slow!

# Fixed parts
SLIDING_WINDOW_SUFFIX = 'Slidingwindow_'

# === CHECK FUNCTIONS ===
def processed_files_exist(pipeline, data_dir="data/processed/"):
    """
    Purpose: check that all expected processed .npy files exist for a pipeline.
    Inputs: pipeline (str), data_dir (str)
    Output: bool (True if all required files are present)
    Notes: checks both train/val/test X and Y files with expected naming convention.
    """
    required_files = [
        f"X_train_LPPT_{pipeline}.npy",
        f"Y_train_LPPT_{pipeline}.npy",
        f"X_val_LPPT_{pipeline}.npy",
        f"Y_val_LPPT_{pipeline}.npy",
        f"X_test_LPPT_{pipeline}.npy",
        f"Y_test_LPPT_{pipeline}.npy"
    ]
    return all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)

def model_exists(pipeline, artifacts_dir="artifacts"):
    """
    Purpose: check whether a saved model checkpoint exists for pipeline.
    Inputs: pipeline (str), artifacts_dir (str)
    Output: bool
    Notes: expects checkpoint at artifacts/LSTM-AE__{pipeline}/model.ckpt
    """
    ckpt_path = os.path.join(artifacts_dir, f"LSTM-AE__{pipeline}", "model.ckpt")
    return os.path.exists(ckpt_path)

def metrics_exist(pipeline, results_dir="results"):
    """
    Purpose: check whether evaluation metrics CSV exists for pipeline.
    Inputs: pipeline (str), results_dir (str)
    Output: bool
    Notes: expects metrics_LSTM-AE__{pipeline}.csv in results_dir
    """
    csv_path = os.path.join(results_dir, f"metrics_LSTM-AE__{pipeline}.csv")
    return os.path.exists(csv_path)

# === RUN ONE PIPELINE ===
def run_pipeline(outlier, norm, lowpass, feat):
    """
    Purpose: run a single pipeline (preprocess -> train -> evaluate) if needed.
    Inputs: configuration choices (outlier, norm, lowpass, feat)
    Output: pipeline name (str)
    Behavior:
      - Skips steps if artifacts already exist (processed files, model, metrics).
      - Sets global variables in preprocess module before running preprocessing.
      - Catches exceptions in training/evaluation and logs error without stopping entire CV.
    """
    pipeline = f"{outlier}_{norm}_{lowpass}_{feat}_{SLIDING_WINDOW_SUFFIX}"
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE: {pipeline}")
    print(f"{'='*60}")

    # --- 1. PREPROCESSING ---
    if processed_files_exist(pipeline):
        print(f"  [SKIP] Preprocessing: .npy files already exist")
    else:
        print(f"  [RUN] Preprocessing data...")
        # set selected preprocessing options in imported preprocess module
        preprocess.OUTLIER_METHOD = outlier
        preprocess.NORMALIZE_METHOD = norm
        preprocess.LOWPASS_FILTER = lowpass
        preprocess.FEATURE_SELECTION = feat
        preprocess.preprocess_all_data()
        print(f"  [DONE] Preprocessing completed")

    # --- 2. TRAINING ---
    if model_exists(pipeline):
        print(f"  [SKIP] Training: model.ckpt already exists")
    else:
        print(f"  [RUN] Training LSTM Autoencoder...")
        try:
            # train_autoencoder.train_lstm_ae saves model to artifacts dir
            train_autoencoder.train_lstm_ae(pipeline)
            print(f"  [DONE] Training completed")
        except Exception as e:
            # return pipeline early to continue outer loop while noting failure
            print(f"  [ERROR] Training failed: {e}")
            return pipeline  # Continua con le altre

    # --- 3. EVALUATION ---
    if metrics_exist(pipeline):
        print(f"  [SKIP] Evaluation: metrics CSV already exists")
    else:
        print(f"  [RUN] Evaluating model...")
        try:
            # NOTE: evaluate.evaluate_pipeline writes metrics and per-window CSVs
            evaluate.evaluate_pipeline(pipeline)  # <-- CAMBIA QUI
            print(f"  [DONE] Evaluation completed")
        except Exception as e:
            print(f"  [ERROR] Evaluation failed: {e}")

    return pipeline

# === MAIN CROSS-VALIDATION ===
def run_cross_validation():
    """
    Purpose: run exhaustive cross-validation over parameter grids.
    Inputs: none (uses global parameter lists)
    Output: saves a summary Excel in results/ if any metrics found
    Behavior:
      - Ensures required directories exist.
      - Iterates over all combinations and runs run_pipeline.
      - Collects per-pipeline 'mean_mse' series into a summary Excel.
    """
    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("../artifacts", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    pipelines = []

    # Print total number of combinations to run
    print(f"\nStarting cross-validation over {len(OUTLIER_METHODS)} × {len(NORMALIZE_METHODS)} × "
          f"{len(LOWPASS_FILTERS)} × {len(FEATURE_SELECTIONS)} = "
          f"{len(OUTLIER_METHODS)*len(NORMALIZE_METHODS)*len(LOWPASS_FILTERS)*len(FEATURE_SELECTIONS)} pipelines\n")

    # Loop over all combinations
    for outlier in OUTLIER_METHODS:
        for norm in NORMALIZE_METHODS:
            for lowpass in LOWPASS_FILTERS:
                for feat in FEATURE_SELECTIONS:
                    pipeline = run_pipeline(outlier, norm, lowpass, feat)
                    pipelines.append(pipeline)

    # === SUMMARY ===
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION COMPLETED")
    print(f"{'='*60}")

    results_dir = "results"
    summary_data = {}

    # Collect mean_mse series from each pipeline's metrics CSV (if present)
    for pipeline in pipelines:
        csv_path = os.path.join(results_dir, f"metrics_LSTM-AE__{pipeline}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0)
                # USE: mean_mse per fault (keeps column aligned by test_name index)
                summary_data[pipeline] = df['mean_mse']
            except Exception as e:
                print(f"Warning: Could not read {csv_path}: {e}")
        else:
            print(f"Warning: Metrics file not found: {csv_path}")

    # If we found any metrics, save aggregated Excel summary
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_xlsx = os.path.join(results_dir, "results_summary.xlsx")
        summary_df.to_excel(summary_xlsx)
        print(f"Summary saved to: {summary_xlsx}")
    else:
        print("No metrics found. Summary not created.")

    print(f"\nTotal pipelines processed: {len(pipelines)}")
    print(f"Check results in: {os.path.abspath('results')}")
