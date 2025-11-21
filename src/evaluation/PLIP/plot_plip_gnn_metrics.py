import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Load PLIP GNN metrics
# ---------------------------
metrics_df = pd.read_csv("gnn_plip_metrics.csv")

# Filter the test row
test_row = metrics_df[metrics_df["split"] == "test"].iloc[0]

# Extract relevant metrics
metrics = {
    "Accuracy": test_row["accuracy"],
    "F1 Score": test_row["f1"],
    "Precision": test_row["precision"],
    "Recall": test_row["recall"],
    "ROC-AUC": test_row["roc_auc"],
}

names = list(metrics.keys())
values = list(metrics.values())

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(names, values)

# Add labels on top of bars
for bar, v in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
             ha='center', fontsize=10)

plt.ylim(0, 1.05)
plt.ylabel("Metric Value", fontsize=12)
plt.title("PLIP-GNN Test Performance Metrics", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("plip_gnn_test_metrics.png", dpi=300)
plt.show()
