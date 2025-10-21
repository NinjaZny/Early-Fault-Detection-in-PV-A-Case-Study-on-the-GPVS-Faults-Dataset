# 🧩 ISCF Project – Meeting Summary and Unified Workflow 21/10

## **Topic**
Build a unified evaluation pipeline and complete the LSTM-AE baseline.

## **Objective (Before Next Meeting)**
Establish a working **general evaluation pipeline**  
(preprocess → train → evaluate → visualize)  
where **training uses only F0L (Train/Val)** and **evaluation is conducted on F1L–F7L (fault datasets)**.  
This phase focuses exclusively on the **LSTM-AE model**.

---

## 🗓️ Key Discussion Points

### 1. Focus on LSTM-AE + F0L for Training
- Only **F0L** is used for training and validation (split sequentially).  
- No testing on F0L itself.

### 2. Unified Fault Evaluation (F1L–F7L)
- Evaluation must be performed on **F1L–F7L** datasets.  
- Each fault type should be tested separately, and results summarized in a single comparison table.

### 3. Task Division & Coordination
- Each team member works on one module (**preprocess**, **train**, **evaluate**, or **visualize**).  
- All modules must share the same **data structure** and **naming conventions**, ensuring full compatibility across scripts.

### 4. Priority: Pipeline Integration
- The main goal before the next meeting is to **run one complete pipeline end-to-end**.  
- Adding new pipelines and models (**GRU-AE**, **CNN-LSTM-AE**, **TCN-AE**, **VAE-AE**) will happen **after** the next meeting.

---

## 🗂️ 0) Recommended Folder Structure
```text
project/
 ├─ data/
 │   ├─ X_train_<pipeline>.npy         # from F0L
 │   ├─ X_val_<pipeline>.npy           # from F0L
 │   └─ test/                          # fault datasets
 │       ├─ X_test_F1L_<pipeline>.npy
 │       ├─ X_test_F2L_<pipeline>.npy
 │       ├─ ...
 │       └─ X_test_F7L_<pipeline>.npy
 ├─ artifacts/     # trained models and logs: LSTM-AE__<pipeline>/
 ├─ results/       # figures and performance summaries (results_summary.xlsx)
 └─ src/
     ├─ preprocess.py
     ├─ train_autoencoder.py
     ├─ evaluate.py
     ├─ visualize.py
     └─ main.py

```

## 1) preprocess.py – Multiple Preprocessing Pipelines

### **Task Description**
- Generate three sets of data for each pipeline:  
  1. `X_train_<pipeline>.npy` – training data (F0L)  
  2. `X_val_<pipeline>.npy` – validation data (F0L)  
  3. `data/test/X_test_F{1..7}L_<pipeline>.npy` – testing data (F1L–F7L)
- **Pipeline example:**  
  Outlier handling (**IQR / MAD / iForest**) → Normalization (**Z-score / Robust**) → Sliding windows (e.g., 200/100) → (Optional) PCA.
- **Important:**  
  All preprocessing steps and parameters (normalization, window length, stride, etc.) must remain **identical** between F0L and F1L–F7L.

### **Outputs**
```text
data/X_train_<pipeline>.npy
data/X_val_<pipeline>.npy
data/test/X_test_F{1..7}L_<pipeline>.npy
```

### **Example Call**
```python
pipeline = "mad_z_win200_s100"
preprocess_data(pipeline, raw_dir="data_raw/", out_dir="data/")
```





## 2) train_autoencoder.py – Train by Pipeline

### **Task Description**

Load X_train_<pipeline>.npy and X_val_<pipeline>.npy from F0L.

Train an LSTM-AE model and save its outputs for later evaluation.

Apply early stopping based on validation loss

### **Outputs(example)**
```python
artifacts/LSTM-AE__<pipeline>/
 ├─ model.ckpt
 ├─ training_log.csv
 └─ (optional) val_errors.npy
```

### **Example Call**
```python
pipeline = "mad_z_win200_s100"
train_lstm_ae(pipeline, data_dir="data/", artifacts_dir="artifacts/")
```


## 3) evaluate.py – Unified Evaluation##
### **Task Description**

Load the trained model from artifacts/LSTM-AE__<pipeline>/model.ckpt.

Evaluate reconstruction results on F1L–F7L datasets (X_test_F1L … X_test_F7L).

Compute reconstruction errors for each fault type and summarize results across all fault scenarios.

### **Inputs & Outputs**
```python
Inputs:
 ├─ artifacts/LSTM-AE__<pipeline>/model.ckpt      # trained model weights
 ├─ data/test/X_test_FkL_<pipeline>.npy           # fault test data (k = 1..7)

Outputs:
 ├─ results/metrics_LSTM-AE__<pipeline>.csv       # metrics per fault type
 └─ results/results_summary.xlsx                  # combined summary across pipelines
```
### **Example call**
```python
pipeline = "mad_z_win200_s100"
evaluate_lstm_ae(
    pipeline,
    data_dir="data/",
    artifacts_dir="artifacts/",
    results_dir="results/"
)
```
## 4) visualize.py – Unified Visualization
### **Task Description**

Plot selected examples (e.g., F1L, F3L, F7L): original vs reconstructed signals.

Plot reconstruction error curves with anomaly points highlighted.

Generate comparison plots between different pipelines and save them for reporting.

### **Outputs**
```python
results/figs/LSTM-AE__<pipeline>/
```

### **Example Call**
```python
pipeline = "mad_z_win200_s100"
plot_test_curves(pipeline, data_dir="data/", artifacts_dir="artifacts/", results_dir="results/")
```

## 5) main.py – Central Controller (Full Workflow)
### **Task Description**

Define which pipelines to run (for this phase, only one LSTM-AE pipeline).

Automatically execute the following sequence:

Preprocessing (skip if files already exist)

Training (F0L only)

Evaluation (F1L–F7L)

Visualization

Ensure the entire workflow can run end-to-end with a single command.

### **Example Main Flow**
```python
PIPELINES = ["mad_z_win200_s100"]

for p in PIPELINES:
    preprocess_data(p, raw_dir="data_raw/", out_dir="data/")                         # produce Train/Val (F0L) + Test (F1L–F7L)
    train_lstm_ae(p, data_dir="data/", artifacts_dir="artifacts/")                  # train only on F0L
    evaluate_lstm_ae(p, data_dir="data/", artifacts_dir="artifacts/", results_dir="results/")  # test on F1L–F7L
    plot_test_curves(p, data_dir="data/", artifacts_dir="artifacts/", results_dir="results/")  # visualize results
```

## Deliverables Before the Next Meeting

One functional preprocessing pipeline that produces:
```

X_train_<pipeline>.npy, X_val_<pipeline>.npy (from F0L)

X_test_F1L–F7L_<pipeline>.npy (for fault testing)

A trained LSTM-AE model saved in artifacts/.

Evaluation results for F1L–F7L saved in .csv and .xlsx formats.

Visualizations (reconstruction and error plots) stored in results/figs/.

main.py can execute the full workflow from start to finish.