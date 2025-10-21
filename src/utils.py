import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sns


def read_mat_file(path, label=None):
    """
    Read .mat (v7.3 HDF5) style and return pandas DataFrame with columns names extracted.
    If label is provided, add a 'label' column with that string.
    """
    with h5py.File(path, 'r') as f:
        # find the '#refs#' group style used in these MAT files
        if '#refs#' not in f:
            # fallback: try to load numeric dataset heuristically (not expected here)
            raise ValueError("Expected '#refs#' group not found in file")
        refs = f['#refs#']
        # keys 'd'..'q' correspond to the 14 columns in the dataset according to your friend
        keys = [k for k in refs.keys() if ('d' <= k <= 'q')]
        if len(keys) == 0:
            raise ValueError("No expected data keys (d..q) found under '#refs#'")
        # read arrays and ravel to 1D
        data_arrays = [np.array(refs[k]).ravel() for k in keys]
        data = np.vstack(data_arrays).T  # shape (n_samples, n_columns)

        # extract human-readable names from '#refs#/v' (UTF-16)
        names = []
        for ref in refs['v']:
            # each ref is an object reference; pick the first element as the dataset
            obj = f[ref[0]]
            try:
                s = obj[:].tobytes().decode('utf-16', errors='ignore').strip('\x00')
            except Exception:
                # fallback to ascii decode if utf-16 fails
                s = obj[:].tobytes().decode('utf-8', errors='ignore').strip('\x00')
            names.append(s)
        # If number of extracted names > data columns, trim; if fewer, create generic names
        if len(names) >= data.shape[1]:
            colnames = names[:data.shape[1]]
        else:
            # fallback generic names D..Q
            colnames = [f"col_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=colnames)
        if label is not None:
            df["label"] = label
        return df
    
def compare_data(df, df_processed, sensors, idx=range(0, 1000), nrows=5, ncols=3, figsize=(14, 7)):
    cols = [c for c in sensors if c in df.columns and c in df_processed.columns]
    if not cols:
        raise ValueError("No common sensor columns found for comparison.")

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    max_plots = nrows * ncols
    plot_cols = cols[:max_plots]

    line_real = line_proc = None
    for j, col in enumerate(plot_cols):
        ax = axes[j]
        y_orig = df[col].values
        y_proc = df_processed[col].values

        if idx is not None:
            y_orig = y_orig[idx]
            y_proc = y_proc[idx]

        line_real, = ax.plot(y_orig, '-', lw=1, label='Original signal')
        line_proc, = ax.plot(y_proc, '--', lw=1, label='Processed signal')
        ax.set_title(col, fontsize=10)
        ax.set_xlim(0, max(len(y_orig), len(y_proc)) - 1)

    for k in range(len(plot_cols), max_plots):
        fig.delaxes(axes[k])

    if line_real is not None and line_proc is not None:
        fig.legend(handles=[line_real, line_proc],
                labels=['Original signal', 'Processed signal'],
                loc='lower center',
                ncol=2,
                fontsize=12)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("temp-compare_signals.png", dpi=200)
    return fig, axes
