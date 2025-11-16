import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

RESULTS_DIR = "results"
ARTIFACTS_DIR = "artifacts"


def find_pipelines():
    # Find all pipeline/model names based on artifacts directory structure
    pattern = os.path.join(ARTIFACTS_DIR, "LSTM-AE__*")
    return [os.path.basename(p).replace("LSTM-AE__", "") for p in glob.glob(pattern) if os.path.isdir(p)]


def pipeline_cv_results_path(pipeline: str) -> str:
    return os.path.join(ARTIFACTS_DIR, f"LSTM-AE__{pipeline}", "cv_results.csv")


def ensure_output_dir(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def plot_cv_summary(df: pd.DataFrame, pipeline: str, out_dir: str):
    """Create a concise set of summary plots useful for hyperparameter analysis.

    Plots created:
      - Correlation heatmap between numeric hyperparameters and mean_val_loss
      - Boxplots of mean_val_loss grouped by learning_rate and by enc1_dim
      - Scatter of latent_dim vs mean_val_loss colored by enc2_dim
      - Parallel coordinates showing hyperparameter patterns for low/med/high loss groups
      - Pairplot (if small) of hyperparameters vs mean_val_loss
    """
    sns.set(style="whitegrid")

    # Ensure numeric conversion
    numeric_cols = [c for c in ["enc1_dim", "enc2_dim", "latent_dim", "learning_rate", "batch_size", "mean_val_loss"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    base_name = pipeline.replace(os.sep, "_")

    # 1) Correlation heatmap
    corr_cols = numeric_cols
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Correlation — {pipeline}")
        path = os.path.join(out_dir, f"{base_name}__corr_heatmap.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"    Saved correlation heatmap -> {path}")

    # 2) Boxplots: mean_val_loss by learning_rate and enc1_dim
    if "learning_rate" in df.columns and "mean_val_loss" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="learning_rate", y="mean_val_loss", palette="Set2")
        plt.title(f"mean_val_loss by learning_rate — {pipeline}")
        path = os.path.join(out_dir, f"{base_name}__loss_by_lr_box.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"    Saved boxplot loss by learning_rate -> {path}")

    if "enc1_dim" in df.columns and "mean_val_loss" in df.columns:
        plt.figure(figsize=(8, 4))
        order = sorted(df["enc1_dim"].unique())
        sns.boxplot(data=df, x="enc1_dim", y="mean_val_loss", order=order, palette="Set3")
        plt.title(f"mean_val_loss by enc1_dim — {pipeline}")
        path = os.path.join(out_dir, f"{base_name}__loss_by_enc1_box.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"    Saved boxplot loss by enc1_dim -> {path}")

    # 3) Scatter: latent_dim vs mean_val_loss colored by enc2_dim (categorical hue)
    if {"latent_dim", "mean_val_loss", "enc2_dim"}.issubset(df.columns):
        plt.figure(figsize=(7, 5))
        # treat enc2_dim as categorical hue
        sns.scatterplot(data=df, x="latent_dim", y="mean_val_loss", hue=df["enc2_dim"].astype(str), palette="viridis", s=80)
        plt.legend(title="enc2_dim", bbox_to_anchor=(1.05, 1), loc=2)
        plt.title(f"latent_dim vs mean_val_loss (enc2_dim hue) — {pipeline}")
        path = os.path.join(out_dir, f"{base_name}__latent_vs_loss_enc2.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"    Saved scatter latent vs loss -> {path}")

    # 4) Parallel coordinates: bin mean_val_loss into 3 groups
    if set(["enc1_dim", "enc2_dim", "latent_dim", "learning_rate", "batch_size", "mean_val_loss"]).issubset(df.columns):
        pc_df = df[["enc1_dim", "enc2_dim", "latent_dim", "learning_rate", "batch_size", "mean_val_loss"]].dropna().copy()
        # normalize numeric features for better visualization
        feature_cols = ["enc1_dim", "enc2_dim", "latent_dim", "learning_rate", "batch_size"]
        for c in feature_cols:
            # scale to 0-1
            minv = pc_df[c].min()
            maxv = pc_df[c].max()
            if maxv > minv:
                pc_df[c] = (pc_df[c] - minv) / (maxv - minv)
        # create loss group
        pc_df["loss_group"] = pd.qcut(pc_df["mean_val_loss"], q=3, labels=["low", "mid", "high"])
        plt.figure(figsize=(9, 5))
        parallel_coordinates(pc_df[[*feature_cols, "loss_group"]], class_column="loss_group", colormap=plt.get_cmap("Set1"))
        plt.title(f"Parallel coordinates (hyperparams) — {pipeline}")
        path = os.path.join(out_dir, f"{base_name}__parallel_coords.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"    Saved parallel coordinates -> {path}")

    # 5) Pairplot for small datasets
    if df.shape[0] <= 200 and len(numeric_cols) >= 2:
        try:
            pp = sns.pairplot(df[numeric_cols], diag_kind="kde", plot_kws={"s": 30})
            path = os.path.join(out_dir, f"{base_name}__pairplot.png")
            pp.fig.suptitle(f"Pairplot — {pipeline}", y=1.02)
            pp.fig.tight_layout()
            pp.savefig(path)
            plt.close()
            print(f"    Saved pairplot -> {path}")
        except Exception as e:
            print(f"    Skipped pairplot due to: {e}")


def plot_comparison_across_pipelines(pipelines: list, out_dir: str):
    """
    For each pipeline collect best mean_val_loss and plot a comparison bar chart and
    additional comparative analyses (distribution of best scores, histogram).
    """
    bests = []
    all_best_values = []
    for p in pipelines:
        path = pipeline_cv_results_path(p)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if "mean_val_loss" not in df.columns:
            continue
        best = df["mean_val_loss"].min()
        bests.append({"pipeline": p, "best_mean_val_loss": best})
        all_best_values.append(best)

    if not bests:
        print("No CV results found to create cross-pipeline comparison.")
        return

    comp_df = pd.DataFrame(bests).sort_values("best_mean_val_loss")
    plt.figure(figsize=(10, max(4, len(comp_df) * 0.25)))
    sns.barplot(data=comp_df, x="best_mean_val_loss", y="pipeline", palette="coolwarm")
    plt.title("Best mean_val_loss per pipeline")
    plt.xlabel("best mean_val_loss")
    plt.ylabel("pipeline")
    cmp_path = os.path.join(out_dir, "cross_pipeline_best_mean_val_loss.png")
    plt.tight_layout()
    plt.savefig(cmp_path)
    plt.close()
    print(f"Saved cross-pipeline comparison -> {cmp_path}")

    # Histogram of best scores across pipelines
    plt.figure(figsize=(6, 4))
    sns.histplot(all_best_values, bins=20, kde=True)
    plt.title("Distribution of best mean_val_loss across pipelines")
    plt.xlabel("best mean_val_loss")
    path = os.path.join(out_dir, "cross_pipeline_best_distribution.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved cross-pipeline best distribution -> {path}")




def plot_pipeline_cv_results(pipeline: str, out_dir: str):
    """
    Read cv_results.csv for the given pipeline and create multiple plots:
      - Heatmap of mean_val_loss (enc1_dim x enc2_dim)
      - Scatter plot of latent_dim vs mean_val_loss (hue=learning_rate)
      - Bar plot of top configs by mean_val_loss

    Saved PNGs are placed in out_dir and named with the pipeline.
    """
    path = pipeline_cv_results_path(pipeline)
    if not os.path.exists(path):
        print(f"  No cv_results.csv found for pipeline {pipeline} at {path}")
        return None

    df = pd.read_csv(path)
    # ensure numeric types
    for c in ["enc1_dim", "enc2_dim", "latent_dim", "learning_rate", "batch_size", "mean_val_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"  Loaded {len(df)} CV rows for {pipeline}")

    # common plot settings
    sns.set(style="whitegrid")

    # 1) Heatmap: pivot enc1_dim x enc2_dim -> mean of mean_val_loss
    if {"enc1_dim", "enc2_dim", "mean_val_loss"}.issubset(df.columns):
        pivot = df.pivot_table(index="enc1_dim", columns="enc2_dim", values="mean_val_loss", aggfunc="mean")
        # sort index/columns numerically if possible
        try:
            pivot = pivot.sort_index(ascending=False)
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        except Exception:
            pass

        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
        plt.title(f"Heatmap mean_val_loss — {pipeline}")
        plt.ylabel("enc1_dim")
        plt.xlabel("enc2_dim")
        heatmap_path = os.path.join(out_dir, f"{pipeline}__cv_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        print(f"    Saved heatmap -> {heatmap_path}")

    # 2) Scatter: latent_dim vs mean_val_loss, hue=learning_rate, size=enc1_dim
    if {"latent_dim", "mean_val_loss", "learning_rate", "enc1_dim"}.issubset(df.columns):
        plt.figure(figsize=(8, 6))
        # Create a small readable marker size mapping
        sizes = (df["enc1_dim"] / df["enc1_dim"].max() * 200) + 20
        scatter = sns.scatterplot(
            data=df,
            x="latent_dim",
            y="mean_val_loss",
            hue="learning_rate",
            size=sizes,
            palette="viridis",
            legend="full",
            sizes=(20, 300),
        )
        plt.title(f"latent_dim vs mean_val_loss — {pipeline}")
        plt.xlabel("latent_dim")
        plt.ylabel("mean_val_loss")
        plt.legend(title="learning_rate", bbox_to_anchor=(1.05, 1), loc=2)
        scatter_path = os.path.join(out_dir, f"{pipeline}__latent_scatter.png")
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        print(f"    Saved scatter -> {scatter_path}")

    # 3) Bar: top N (best) configurations by mean_val_loss
    if "mean_val_loss" in df.columns:
        df_sorted = df.sort_values("mean_val_loss").reset_index(drop=True)
        topn = 12 if len(df_sorted) >= 12 else len(df_sorted)
        df_top = df_sorted.head(topn).copy()
        # compact label for each config
        def make_label(row):
            return f"e1({int(row['enc1_dim'])})_e2({int(row['enc2_dim'])})_l({int(row['latent_dim'])})_lr({row['learning_rate']})"

        df_top["config"] = df_top.apply(make_label, axis=1)
        plt.figure(figsize=(max(8, topn * 0.8), 6))
        sns.barplot(data=df_top, x="config", y="mean_val_loss", palette="magma")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top {topn} CV configs by mean_val_loss — {pipeline}")
        plt.xlabel("config (enc1_enc2_latent_lr)")
        plt.ylabel("mean_val_loss")
        bar_path = os.path.join(out_dir, f"{pipeline}__top_configs.png")
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()
        print(f"    Saved top-configs bar -> {bar_path}")

    return df




def main():
    pipelines = find_pipelines()
    if not pipelines:
        print("No pipelines found in artifacts/. Please run evaluation first.")
        return

    print(f"Found pipelines: {pipelines}")

    out_dir = os.path.join(os.getcwd(), "visualise")
    ensure_output_dir(out_dir)

    # For each pipeline build and save aggregated hyperparameter analysis plots
    processed = 0
    for pipeline in pipelines:
        path = pipeline_cv_results_path(pipeline)
        plot_pipeline_cv_results(pipeline, out_dir)
        if not os.path.exists(path):
            continue
        print(f"\nProcessing pipeline: {pipeline}")
        df = pd.read_csv(path)
        if df is None or df.shape[0] == 0:
            print(f"  Empty or unreadable cv_results for {pipeline}")
            continue
        # Create aggregated analysis plots (more useful than per-sample visualizations)
        plot_cv_summary(df, pipeline, out_dir)
        processed += 1

    # Cross-pipeline comparison
    plot_comparison_across_pipelines(pipelines, out_dir)

    if processed == 0:
        print("No cv_results.csv found to plot.")
    else:
        print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()