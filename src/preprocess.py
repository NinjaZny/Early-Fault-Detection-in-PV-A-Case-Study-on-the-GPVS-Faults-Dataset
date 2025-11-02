import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import utils
import preprocess_methods

OUTLIER_METHOD = 'none' # options: 'none', 'IQR'
NORMALIZE_METHOD = 'minmax' # 'none', 'zscore', 'minmax', 'robust'
LOWPASS_FILTER = 'butterworth' # 'none', 'butterworth', 'moving_average'
FEATURE_SELECTION = 'none' # 'none', 'pca', 'robustpca', 'kernelpca'

RANDOM_SEED = 42

OUTLIER_WINDOW = 11

CUTOFF = 300
FS = 10000
BUTTER_ORDER = 2
MA_WINDOW = 7

PCA_COMPONENTS = 8

WINDOW_LEN = 200
STRIDE = 15


def _save_or_append(array: np.ndarray, path: str):
    """if path exists, load and append along axis 0; else just save."""
    if os.path.exists(path):
        old = np.load(path, allow_pickle=False)
        try:
            array = np.concatenate([old, array], axis=0)
        except Exception as e:
            raise ValueError(
                f"Concatenation failed for {path}. Old shape={old.shape}, new shape={array.shape}"
            ) from e
    np.save(path, array)

def _split_indices(n: int, seed: int, ratios=(0.8, 0.1, 0.1)):
    """Return split indices for train/val/test with fixed seed."""
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    idx_train = order[:n_train]
    idx_val   = order[n_train:n_train+n_val]
    idx_test  = order[n_train+n_val:]
    return idx_train, idx_val, idx_test

def save_processed_dataset(X: np.ndarray, Y: np.ndarray, label_str: str,
                           processed_dir: str = "data/processed"):
    if label_str.endswith("L"):
        group = "LPPT"
    elif label_str.endswith("M"):
        group = "MPPT"
    else:
        group = "Unknown"

    # Construct filename suffix (inline as requested)
    tail = f"{OUTLIER_METHOD}_{NORMALIZE_METHOD}_{LOWPASS_FILTER}_{FEATURE_SELECTION}_Slidingwindow_.npy"

    # === VALIDATE Y SHAPE ===
    if Y.ndim == 0:
        Y = np.full((X.shape[0],), Y)
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Y shape {Y.shape} does not match X {X.shape}")

    if label_str.startswith("F0"):
        idx_tr, idx_va, idx_te = _split_indices(X.shape[0], RANDOM_SEED, (0.8, 0.1, 0.1))

        paths = {
            "train": (f"X_train_{group}_{tail}", f"Y_train_{group}_{tail}"),
            "val": (f"X_val_{group}_{tail}", f"Y_val_{group}_{tail}"),
            "test": (f"X_test_{group}_{tail}", f"Y_test_{group}_{tail}"),
        }

        for split, idx in (("train", idx_tr), ("val", idx_va), ("test", idx_te)):
            x_path = os.path.join(processed_dir, paths[split][0])
            y_path = os.path.join(processed_dir, paths[split][1])
            _save_or_append(X[idx], x_path)
            _save_or_append(Y[idx], y_path)

        print(f"[save] F0 {group}: {X.shape[0]} samples → train/val/test split")

    else:
        # FAULT DATA: SAVE EACH FAULT IN A SEPARATE FILE
        # OLD CODE (merged all faults — REMOVED):
        # x_path = os.path.join(processed_dir, f"X_test_{group}_{tail}")
        # _save_or_append(X, x_path)
        #
        # WHY REMOVED?
        # - Mixing F1L + F2L → cannot evaluate per-fault performance
        # - Cannot compute lead time or early detection


        #One file per fault → e.g., X_test_F1L_LPPT_...
        x_path = os.path.join(processed_dir, f"X_test_{label_str}_{group}_{tail}")
        y_path = os.path.join(processed_dir, f"Y_test_{label_str}_{group}_{tail}")

        # Save directly (no append) → each fault = one file
        np.save(x_path, X)
        np.save(y_path, Y)

        print(f"[save] FAULT {label_str} ({group}): {X.shape} → {os.path.basename(x_path)}")


def preprocess_data(df, columns, sensors,
                    outlier_method = OUTLIER_METHOD, 
                    normalize_method = NORMALIZE_METHOD, 
                    lowpass_filter = LOWPASS_FILTER, 
                    feature_selection = FEATURE_SELECTION,
                    pca_map = None
                    ):
    df_corrected = df.copy()
    
    print(f"Starting preprocessing: {df['label'][0]}")

    # Outlier removal
    if outlier_method == 'IQR':
        for col in sensors:
            if col == 'label':
                continue
            x = df_corrected[col].values
            x_replaced, mask = preprocess_methods.replace_outliers_local_mean(x, window=OUTLIER_WINDOW)
            df_corrected[col] = x_replaced
    elif outlier_method == 'none':
        pass
    else:
        raise ValueError(f"Unknown outlier method: {outlier_method}")
    # utils.compare_data(df, df_corrected, sensors, range(0, 1000), f"oulier-removed_{outlier_method}_{df['label'][0]}")
    utils.compare_data(df, df_corrected, sensors, plotname = f"oulier-removed_{df['label'][0]}")

    # Normalization
    if normalize_method == 'zscore':
        df_normalized = preprocess_methods.zscore_dataframe(df_corrected, sensors)
    elif normalize_method == 'minmax':
        df_normalized = preprocess_methods.minmax_dataframe(df_corrected, sensors)
    elif normalize_method == 'robust':
        df_normalized = preprocess_methods.robust_dataframe(df_corrected, sensors)
    elif normalize_method == 'none':
        df_normalized = df_corrected
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")
    # utils.compare_data(df, df_normalized, sensors, range(0, 1000), f"normalized_{normalize_method}_{df['label'][0]}")
    utils.compare_data(df, df_normalized, sensors, plotname = f"normalized_{df['label'][0]}")
    
    # Low-pass filtering
    if lowpass_filter == 'butterworth':
        df_filtered = preprocess_methods.butterworth_lowpass_filter_dataframe(df_normalized, sensors, cutoff=CUTOFF, fs=FS, order=BUTTER_ORDER)
    elif lowpass_filter == 'moving_average':
        df_filtered = preprocess_methods.moving_average_filter_dataframe(df_normalized, sensors, window=MA_WINDOW, center=True, min_periods=1)
    elif lowpass_filter == 'none':
        df_filtered = df_normalized
    else:
        raise ValueError(f"Unknown low-pass filter method: {lowpass_filter}")
    # utils.compare_data(df, df_filtered, sensors, range(0, 1000), f"filtered_{lowpass_filter}_{df['label'][0]}")
    utils.compare_data(df, df_filtered, sensors, plotname = f"filtered_{df['label'][0]}")

    # Feature selection
    if feature_selection == 'pca' or feature_selection == 'robustpca' or feature_selection == 'kernelpca':
        df_extracted, pca, n_comp = preprocess_methods.apply_pca_safe(df_filtered, sensors, feature_selection, n_components_requested=PCA_COMPONENTS, label_str=df['label'][0], pca_map=pca_map)
        utils.compare_data_pca(df, df_extracted, sensors, n_comp, plotname = f"feature-selection_{df['label'][0]}")
    elif feature_selection == 'none':
        df_extracted = df_filtered
        pca = None
    else:
        raise ValueError(f"Unknown feature selection method: {feature_selection}")

    # Ensure all original sensor columns are present in processed DataFrame
    if feature_selection == 'none':
        for col in sensors:
            if col in df.columns:
                df_extracted[col] = df[col]

    # Sliding window
    X, Y = preprocess_methods.sliding_windows(df_extracted, window_len=WINDOW_LEN, stride=STRIDE)
    
    return X, Y, pca


def preprocess_all_data():
    dataset_folder = '../data/GPVS-Faults'
    processed_data_folder = 'data/processed'
    plot_folder = 'data/plot'

    os.makedirs(processed_data_folder, exist_ok=True)
    # delete all files in processed_data_folder
    for f in os.listdir(processed_data_folder):
        fp = os.path.join(processed_data_folder, f)
        if os.path.isfile(fp):
            os.remove(fp)

    os.makedirs(plot_folder, exist_ok=True)

    filenames = sorted([f for f in os.listdir(dataset_folder) if f.endswith('.mat')])

    # create class_names, like "F0L"
    class_names = [fn[:3] for fn in filenames]
    le = LabelEncoder()
    le.fit(class_names)

    columns = ["Time","Ipv","Vpv","Vdc","ia","ib","ic","va","vb","vc","Iabc","If","Vabc","Vf","label"]

    pca_map = {}

    for fn in filenames:
        label_str = fn[:3]
        if not (label_str.startswith("F0") and label_str[-1] in ("L", "M")):
            continue
        print(f"Processing file: {fn}")
        path = os.path.join(dataset_folder, fn)
        df = utils.read_mat_file(path, label=label_str)

        sensors = [c for c in columns if c in df.columns and c != 'Time' and c != 'label']
        X, Y, pca = preprocess_data(df, columns, sensors, pca_map=None)

        key = "LPPT" if label_str.endswith("L") else "MPPT"
        pca_map[key] = pca  # store PCA model for later

        save_processed_dataset(X, Y, label_str)

    for fn in filenames:
        label_str = fn[:3]
        if label_str.startswith("F0"):
            continue
        print(f"Processing fault file: {fn}")
        path = os.path.join(dataset_folder, fn)
        df = utils.read_mat_file(path, label=label_str)
        sensors = [c for c in columns if c in df.columns and c != 'Time' and c != 'label']
        key = "LPPT" if label_str.endswith("L") else "MPPT"
        X, Y, _ = preprocess_data(df, columns, sensors, pca_map=pca_map.get(key))
        save_processed_dataset(X, Y, label_str)  # ← Saves F1L, F2M, etc. separately

    for f in os.listdir(processed_data_folder):
        if f.endswith(".npy"):
            arr = np.load(os.path.join(processed_data_folder, f))
            print(f, arr.shape)
