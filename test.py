# test.py — minimal env check (no project imports)

import sys

print("=== Environment Check ===")
print("Python version:", sys.version)

# ---- Test package imports ----
try:
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from scipy.signal import butter, filtfilt
    import torch
    import yaml
    print("✅ All main libraries imported successfully!")
except Exception as e:
    print("❌ Import error:", e)

# ---- Torch & CUDA info ----
try:
    import torch
    print("Torch version:", torch.__version__)
    print("Torch CUDA version (build):", torch.version.cuda)  # None => CPU build
    print("CUDA available via torch:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("❌ Torch error:", e)
