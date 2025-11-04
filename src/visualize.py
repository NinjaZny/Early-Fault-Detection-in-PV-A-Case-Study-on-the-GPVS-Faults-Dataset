
import os

import glob

RESULTS_DIR = "results"
ARTIFACTS_DIR = "artifacts"

def find_pipelines():
    # Find all pipeline/model names based on artifacts directory structure
    pattern = os.path.join(ARTIFACTS_DIR, "LSTM-AE__*")
    return [os.path.basename(p).replace("LSTM-AE__", "") for p in glob.glob(pattern) if os.path.isdir(p)]

def find_per_sample_error_files(pipeline):
    pattern = os.path.join(RESULTS_DIR, f"per_sample_errors_LSTM-AE__{pipeline}_*.csv")
    return glob.glob(pattern)

def find_metrics_file(pipeline):
    return os.path.join(RESULTS_DIR, f"metrics_LSTM-AE__{pipeline}.csv")

def find_metadata_file(pipeline):
    return os.path.join(RESULTS_DIR, f"metadata_LSTM-AE__{pipeline}.json")

def main():
    # Discover available pipelines/models
    pipelines = find_pipelines()
    if not pipelines:
        print("No pipelines found in artifacts/. Please run evaluation first.")
        return

    print(f"Found pipelines: {pipelines}")
    for pipeline in pipelines:
        per_sample_files = find_per_sample_error_files(pipeline)
        metrics_file = find_metrics_file(pipeline)
        metadata_file = find_metadata_file(pipeline)
        print(f"\nPipeline: {pipeline}")
        print(f"  Per-sample error files: {per_sample_files if per_sample_files else 'None found'}")
        print(f"  Metrics file: {metrics_file if os.path.exists(metrics_file) else 'Not found'}")
        print(f"  Metadata file: {metadata_file if os.path.exists(metadata_file) else 'Not found'}")
        plot_reconstruction_and_error_curves(pipeline)
        plot_confusion_matrix_and_roc(pipeline)
    plot_comparison_charts(pipelines)

def plot_reconstruction_and_error_curves(pipeline):
    """
    Plot reconstructed vs. original signals and error + threshold curves for a given pipeline.
    """
    print(f"[{pipeline}] Plotting reconstructed vs. original signals and error + threshold curves...")
    # Check for required files (test data, model checkpoint, metadata)
    # This is a stub: actual implementation will require .npy test data and model checkpoint
    # For now, just print what would be required
    model_dir = os.path.join(ARTIFACTS_DIR, f"LSTM-AE__{pipeline}")
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    print(f"  Would need: {ckpt_path}, X_test_...npy, and metadata for {pipeline}")
    # TODO: Implement actual plotting when data is available

def plot_confusion_matrix_and_roc(pipeline):
    """
    Generate confusion matrix and ROC curve for a given pipeline.
    """
    print(f"[{pipeline}] Plotting confusion matrix and ROC curve...")
    # Would need per-sample error CSVs, threshold, and ground truth labels
    print(f"  Would need: per_sample_errors_LSTM-AE__{pipeline}_*.csv, threshold, and ground truth labels")
    # TODO: Implement actual plotting when data is available

def plot_comparison_charts(pipelines):
    """
    Create comparison charts across pipelines × models.
    """
    print("[ALL] Plotting comparison charts across pipelines × models...")
    # Would need metrics_LSTM-AE__{pipeline}.csv for all pipelines
    print("  Would need: metrics_LSTM-AE__{pipeline}.csv for all pipelines")
    # TODO: Implement actual plotting when data is available

if __name__ == "__main__":
    main()
