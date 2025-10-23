import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from scipy.signal import butter, sosfiltfilt


def detect_outliers(x, k=1.5):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x < lower) | (x > upper)

def replace_outliers_local_mean(x, window=11):
    """
    Replace outlier points with centered rolling mean.
    Returns replaced array and boolean mask of outliers.
    """
    x = np.asarray(x, dtype=float).copy()
    mask = detect_outliers(x)
    if not mask.any():
        return x, mask
    wl = int(window)
    if wl < 3: wl = 3
    if wl % 2 == 0: wl += 1
    ser = pd.Series(x)
    roll = ser.rolling(window=wl, center=True, min_periods=1).mean().values
    x[mask] = roll[mask]
    return x, mask


def zscore_series(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sigma = x.std()
    if sigma < eps:
        return x - mu
    return (x - mu) / sigma

def zscore_dataframe(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = zscore_series(out[c].values)
    return out

def minmax_series(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    xmin = x.min()
    xmax = x.max()
    rng = xmax - xmin
    if rng < eps:
        return x - xmin
    return (x - xmin) / rng

def minmax_dataframe(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = minmax_series(out[c].values)
    return out


def robust_series(x, eps=1e-8):
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    # print(f"robust_series: med={med}, IQR={iqr}")
    if iqr < eps:
        return x - med
    return (x - med) / iqr

def robust_dataframe(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = robust_series(out[c].values)
    return out


def butterworth_lowpass_filter_dataframe(
    df, 
    sensors, 
    cutoff=0.1,
    fs=1.0,
    order=4
):
    if cutoff <= 0 or fs <= 0:
        raise ValueError("cutoff and fs must be positive numbers")
    
    nyq = 0.5 * fs
    wn = cutoff / nyq
    if not (0 < wn < 1):
        raise ValueError(f"Normalized cutoff frequency must be between 0 and 1, got {wn}")

    sos = butter(order, wn, btype='low', output='sos')

    out = df.copy()

    for col in sensors:
        if col not in out.columns:
            continue
        if not np.issubdtype(out[col].dtype, np.number):
            continue

        x = out[col].values
        y = sosfiltfilt(sos, x)
        out[col] = y

    return out


def moving_average_filter_dataframe(
    df,
    sensors,
    window=5,
    center=True,
    min_periods=1
):
    if window is None or int(window) < 1:
        raise ValueError("window must be a positive integer")
    out = df.copy()

    for col in sensors:
        if col not in out.columns:
            continue
        if not np.issubdtype(out[col].dtype, np.number):
            continue
        out[col] = (
            out[col]
            .rolling(window=int(window), center=bool(center), min_periods=int(min_periods))
            .mean()
            .values
        )
    return out



def apply_pca_safe(df, sensors, feature_selection, n_components_requested=4, label_str=None, pca_map=None):
    X = df[sensors].values
    X = np.asarray(X)

    n_samples, n_features = X.shape
    max_comp = min(n_samples, n_features)
    n_comp = max(1, min(int(n_components_requested), max_comp))

    # Reuse PCA model for non-F0 data
    if label_str is not None and not label_str.startswith("F0"):
        print(f"Using PCA model calculated by: F0{label_str[-1]}")
        if label_str.endswith("L"):
            pca_model = pca_map.get("LPPT", None)
        elif label_str.endswith("M"):
            pca_model = pca_map.get("MPPT", None)

        if pca_model is None:
            raise ValueError("pca_model must be provided when label is not starting with 'F0'.")

        Xp = pca_model.transform(X)
        n_comp = Xp.shape[1]
        pca = None

    else:

        if feature_selection == 'pca':
            pca = PCA(n_components=n_comp)
            Xp = pca.fit_transform(X)

        elif feature_selection == 'kernelpca':
            # print("Using KernelPCA with RBF kernel.")
            idx = np.random.choice(X.shape[0], 50000, replace=False) # only choose 50000 samples to fit to avoid memory issue
            pca = KernelPCA(n_components=n_comp, kernel='rbf', gamma=0.1) # hyperparameter kernel, gamma can be tuned
            pca.fit(X[idx])
            Xp = pca.transform(X)

        elif feature_selection == 'robustpca':
            # print("Using RobustPCA.")
            import rpca
            pca = rpca.RobustPCA(n_components=n_comp) # can be tuned: max_iter, tol, gamma, etc.
            pca.fit(X)
            Xp = pca.transform(X)
            n_comp = Xp.shape[1]

    pca_cols = [f"PCA_{i+1}" for i in range(n_comp)]
    Xp_df = pd.DataFrame(Xp, columns=pca_cols, index=df.index)

    return Xp_df, pca, n_comp


def sliding_windows(arr, window_len, stride=None):

    if hasattr(arr, "columns") and hasattr(arr, "to_numpy") and "label" in getattr(arr, "columns", []):
        labels_seq = arr["label"].to_numpy()
        feats = arr.drop(columns=["label"]).to_numpy()
    else:
        a = np.asarray(arr)
        if a.ndim < 2:
            raise ValueError("ndarray must have at least 2 dimensions")
        labels_seq = a[..., -1]
        feats = a[..., :-1]

    feats = np.asarray(feats)
    labels_seq = np.asarray(labels_seq)

    n = feats.shape[0]
    if stride is None:
        stride = max(1, window_len // 2)

    if window_len > n:
        empty_windows = np.empty((0, window_len) + feats.shape[1:], dtype=feats.dtype)
        empty_labels = np.empty((0,) + labels_seq.shape[1:], dtype=labels_seq.dtype)
        return empty_windows, empty_labels

    starts = np.arange(0, n - window_len + 1, stride)

    windows = np.stack([feats[s:s + window_len] for s in starts], axis=0)
    labels = np.stack([labels_seq[s] for s in starts], axis=0)

    return windows, labels