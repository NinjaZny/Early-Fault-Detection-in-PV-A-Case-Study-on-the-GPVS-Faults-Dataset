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
    
def compare_data(df, df_processed, sensors, idx=range(30000, 30200), plotname="plot",
                 nrows=5, ncols=3, figsize=(14, 7)):
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
        ax.set_title(f"{col} ({plotname})", fontsize=10)
        ax.set_xlim(0, max(len(y_orig), len(y_proc)) - 1)

    # remove unused axes
    for k in range(len(plot_cols), max_plots):
        fig.delaxes(axes[k])

    if line_real is not None and line_proc is not None:
        fig.legend(handles=[line_real, line_proc],
                   labels=['Original signal', 'Processed signal'],
                   loc='lower center',
                   ncol=2,
                   fontsize=12)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(f"./data/plot/{plotname}.png", dpi=200)
    return fig, axes


def compare_data_pca(df, df_processed, sensors, n_comp,
                     idx=range(0, 1000), plotname="plot",
                     nrows_sensors=3, ncols_sensors=3,
                     nrows_pca=2, ncols_pca=3,
                     figsize=None,
                     height_ratio_sensors=5, height_ratio_pca=3):

    # Validate/collect columns
    sensor_cols = [c for c in sensors if c in df.columns]
    if not sensor_cols:
        raise ValueError("No valid sensor columns found in df for plotting.")

    pca_cols_all = [f"PCA_{i+1}" for i in range(n_comp)]
    pca_cols = [c for c in pca_cols_all if c in df_processed.columns]
    if not pca_cols:
        raise ValueError("No PCA component columns found in df_processed.")

    # Apply index window
    if idx is not None:
        df_seg = df.loc[idx, sensor_cols]
        dfp_seg = df_processed.loc[idx, pca_cols]
    else:
        df_seg = df[sensor_cols]
        dfp_seg = df_processed[pca_cols]

    # Compute a dynamic figsize if not provided (more height per row to reduce vertical crowding)
    if figsize is None:
        # ~2.2 inches per row is a comfortable default; add margins
        total_rows = nrows_sensors + nrows_pca
        height = 2.2 * total_rows + 2.5
        width = 16
        figsize = (width, height)

    # Build figure using a two-row GridSpec: [Sensors | PCA]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[height_ratio_sensors, height_ratio_pca], hspace=0.4)

    # --- 1) Sensor time series ---
    gs_s = gs[0].subgridspec(nrows_sensors, ncols_sensors, wspace=0.25, hspace=0.5)
    axes_sensors = []
    max_sensors = nrows_sensors * ncols_sensors
    plot_sensor_cols = sensor_cols[:max_sensors]
    for j, col in enumerate(plot_sensor_cols):
        ax = fig.add_subplot(gs_s[j // ncols_sensors, j % ncols_sensors])
        y = df_seg[col].values
        ax.plot(y, lw=1)
        ax.set_title(f"{col} ({plotname})", fontsize=10)
        ax.set_xlim(0, len(y) - 1)
        axes_sensors.append(ax)
    # Remove unused tiles if grid > plotted columns
    for j in range(len(plot_sensor_cols), max_sensors):
        fig.add_subplot(gs_s[j // ncols_sensors, j % ncols_sensors]).remove()

    # --- 2) PCA component time series ---
    gs_p = gs[1].subgridspec(nrows_pca, ncols_pca, wspace=0.25, hspace=0.5)
    axes_pca = []
    max_pca = nrows_pca * ncols_pca
    plot_pca_cols = pca_cols[:max_pca]
    for j, col in enumerate(plot_pca_cols):
        ax = fig.add_subplot(gs_p[j // ncols_pca, j % ncols_pca])
        y = dfp_seg[col].values
        ax.plot(y, lw=1)
        ax.set_title(f"{col} ({plotname})", fontsize=10)
        ax.set_xlim(0, len(y) - 1)
        axes_pca.append(ax)
    for j in range(len(plot_pca_cols), max_pca):
        fig.add_subplot(gs_p[j // ncols_pca, j % ncols_pca]).remove()

    # Titles, layout, save
    fig.suptitle(f"{plotname} — Original Sensors and PCA Components", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(f"./data/plot/{plotname}.png", dpi=200)

    axes_dict = {"sensors": axes_sensors, "pca": axes_pca}
    return fig, axes_dict