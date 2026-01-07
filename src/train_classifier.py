import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import cycle

# Se hai un file config.py, mantieni l'import, altrimenti commentalo
try:
    from config import CFG
except ImportError:
    pass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROCESSED_DIR = "../data/processed"

LABEL_MAP = {
    "F0": 0, "F1": 1, "F2": 2, "F3": 3,
    "F4": 4, "F5": 5, "F6": 6, "F7": 7
}
NUM_CLASSES = len(LABEL_MAP)
CLASS_NAMES = list(LABEL_MAP.keys())


# ==========================
# Definition LSTM
# ==========================
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=8, dropout=0.3):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        out, (h_n, c_n) = self.lstm(x)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits


# ==========================
# Funzioni di Caricamento Dati
# ==========================
def load_and_prepare_data(pipeline_tail, pipeline_suffix="L"):
    group = "LPPT" if pipeline_suffix == "L" else "MPPT"
    all_X, all_Y = [], []

    print(f"[INFO] Load dataset for {group}...")

    # 1. Carica F0 (Sano)
    f0_name = f"X_train_{group}_{pipeline_tail}.npy"
    f0_path = os.path.join(PROCESSED_DIR, f0_name)

    if os.path.exists(f0_path):
        x_f0 = np.load(f0_path)
        if x_f0.ndim != 3: x_f0 = np.transpose(x_f0, (0, 2, 1)) if x_f0.shape[1] < x_f0.shape[2] else x_f0
        y_f0 = np.zeros(len(x_f0), dtype=int)
        all_X.append(x_f0)
        all_Y.append(y_f0)
    else:
        print(f"[WARNING] File F0 non found: {f0_path}")

    # 2. Carica F1-F7 (Guasti)
    for fault_label in ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]:
        full_label = fault_label + pipeline_suffix
        fname = f"X_test_{full_label}_{group}_{pipeline_tail}.npy"
        fpath = os.path.join(PROCESSED_DIR, fname)

        if os.path.exists(fpath):
            x_fault = np.load(fpath)
            if x_fault.ndim != 3: x_fault = np.transpose(x_fault, (0, 2, 1)) if x_fault.shape[1] < x_fault.shape[
                2] else x_fault
            y_fault = np.full(len(x_fault), LABEL_MAP[fault_label], dtype=int)
            all_X.append(x_fault)
            all_Y.append(y_fault)
        else:
            print(f"[WARNING] File not found: {fname}")

    if not all_X:
        raise ValueError("No file has been loaded!")

    return np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0)


# ==========================
# Training Function
# ==========================
def train_classifier(pipeline_name):
    SUFFIX = 'L'  # LPPT
    print(f"\n{'=' * 40}")
    print(f"START CLASSIFICATION LSTM  - Pipeline: {pipeline_name}")
    print(f"{'=' * 40}")

    # 1. Data
    X, Y = load_and_prepare_data(pipeline_name, SUFFIX)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Modello
    input_dim = X.shape[2]
    model = LSTMClassifier(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)

    # Weight
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 5.0, 5.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 50
    best_acc = 0.0
    model_save_path = f"best_classifier_{SUFFIX}.pth"

    train_losses = []
    val_accuracies = []

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch + 1}/{NUM_EPOCHS}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        val_accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f} | Val Acc = {acc:.2f}%")

    print(f"\n[TRAINING COMPLETO] Best Acc: {best_acc:.2f}%")


    # 5. More accurate study:
    print("\n[INFO] Valutazione Approfondita sul Test Set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_preds = []
    all_labels = []

    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(X_test)) * 1000

    # Metrics
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"\n--- Performance Metrics Extra ---")
    print(f"Cohen's Kappa Score: {kappa:.4f} (1.0 = perfect)")
    print(f"Totale Inference time ({len(X_test)} samples): {total_inference_time:.4f} sec")
    print(f"Average Time per Sample: {avg_inference_time_ms:.4f} ms")

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {SUFFIX}')
    plt.savefig(f"confusion_matrix_{SUFFIX}.png")



