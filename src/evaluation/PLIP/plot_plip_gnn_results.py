import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score
)

# ----------------------------------------------------------
# Load prediction-level and metric-level results
# ----------------------------------------------------------
pred_df = pd.read_csv("gnn_plip_test_predictions.csv")
metrics_df = pd.read_csv("gnn_plip_metrics.csv")

y_true = pred_df["true_label"].values
y_prob = pred_df["prob_positive"].values
y_pred = pred_df["pred_label"].values

# ----------------------------------------------------------
# 0. Compute balanced accuracy since CSV does not include it
# ----------------------------------------------------------
bal_acc = balanced_accuracy_score(y_true, y_pred)

# ----------------------------------------------------------
# 1. ROC Curve
# ----------------------------------------------------------
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("PLIP–GNN ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plip_gnn_roc_curve.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 2. Precision–Recall Curve
# ----------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PLIP–GNN Precision–Recall Curve")
plt.tight_layout()
plt.savefig("plip_gnn_pr_curve.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 3. Confusion Matrix
# ----------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(5,5))
disp.plot(colorbar=False)
plt.title("PLIP–GNN Confusion Matrix")
plt.tight_layout()
plt.savefig("plip_gnn_confusion_matrix.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 4. Metrics Bar Chart (including computed bal_acc)
# ----------------------------------------------------------
metric_names = ["accuracy", "f1", "precision", "recall", "bal_acc", "roc_auc"]

test_row = metrics_df[metrics_df["split"] == "test"]

metric_values = [
    float(test_row["accuracy"]),
    float(test_row["f1"]),
    float(test_row["precision"]),
    float(test_row["recall"]),
    bal_acc,
    float(test_row["roc_auc"]),
]

plt.figure(figsize=(8,5))
bars = plt.bar(metric_names, metric_values, color="steelblue")
plt.ylim(0,1)
plt.ylabel("Score")
plt.title("PLIP–GNN Test Performance Metrics")

for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0,3), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("plip_gnn_metrics_barplot.png", dpi=300)
plt.close()

print("Saved PLIP–GNN plots:")
print(" • plip_gnn_roc_curve.png")
print(" • plip_gnn_pr_curve.png")
print(" • plip_gnn_confusion_matrix.png")
print(" • plip_gnn_metrics_barplot.png")
