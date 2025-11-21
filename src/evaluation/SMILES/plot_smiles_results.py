#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ---------------------------------------------------------
# Helper: infer model name from filename
# ---------------------------------------------------------
def infer_model_name(path):
    name = path.lower()
    if "ridge" in name:
        return "Ridge Classifier"
    if "rf" in name or "randomforest" in name:
        return "Random Forest"
    if "mlp" in name or "neural" in name:
        return "MLP"
    return path  # fallback


# ---------------------------------------------------------
# Helper: load CSV with flexible column names
# ---------------------------------------------------------
def load_predictions(path):
    df = pd.read_csv(path)

    # Map flexible names → standard names
    colmap = {
        "true_label": "y_true",
        "label": "y_true",
        "predicted_label": "y_pred",
        "pred_label": "y_pred",
        "pred": "y_pred",
        "pred_prob": "y_prob",
        "prob": "y_prob",
        "prediction_probability": "y_prob",
    }

    # Standardize columns
    std = {}
    for col in df.columns:
        key = col.lower()
        if key in colmap:
            std[colmap[key]] = df[col]

    # If no probabilities are given, generate placeholder
    if "y_prob" not in std:
        print(f"⚠ WARNING: No probabilities found in {path}. Using y_pred as probability.")
        std["y_prob"] = std["y_pred"].astype(float)

    return pd.DataFrame(std)


# ---------------------------------------------------------
# Helper: create all required plots for one model
# ---------------------------------------------------------
def plot_for_model(df, model_name):
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    y_prob = df["y_prob"]

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

    # ---- ROC Curve ----
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(f"{model_name} — ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_roc_curve.png")
    plt.close()

    # ---- Precision–Recall ----
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision)
    plt.title(f"{model_name} — Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_precision_recall.png")
    plt.close()

    print(f"✓ Saved plots for {model_name}")


# ---------------------------------------------------------
# Main script
# ---------------------------------------------------------
if len(sys.argv) < 2:
    print("Usage: python plot_smiles_results.py <csv1> <csv2> ...")
    sys.exit(1)

csv_files = sys.argv[1:]

print(f"Processing {len(csv_files)} files...\n")

for path in csv_files:
    print(f"--- Loading {path} ---")
    df = load_predictions(path)
    model_name = infer_model_name(path)
    plot_for_model(df, model_name)

print("\n🎉 All plots generated!")
