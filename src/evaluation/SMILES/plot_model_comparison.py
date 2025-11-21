import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Load metrics
# -------------------------------------------------
ridge_df = pd.read_csv("ridge_binary_metrics.csv")
rf_df    = pd.read_csv("rf_binary_metrics.csv")
mlp_df   = pd.read_csv("mlp_binary_metrics.csv")

# Extract only the test rows
ridge_test = ridge_df[ridge_df["split"] == "test"].iloc[0]
rf_test    = rf_df[rf_df["split"] == "test"].iloc[0]
mlp_test   = mlp_df[mlp_df["split"] == "test"].iloc[0]

# -------------------------------------------------
# Combine metrics into a single dataframe
# -------------------------------------------------
model_names = ["Ridge", "Random Forest", "MLP"]

metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]

data = pd.DataFrame({
    "Model": model_names,
    "accuracy":  [ridge_test["accuracy"],  rf_test["accuracy"],  mlp_test["accuracy"]],
    "f1":        [ridge_test["f1"],        rf_test["f1"],        mlp_test["f1"]],
    "precision": [ridge_test["precision"], rf_test["precision"], mlp_test["precision"]],
    "recall":    [ridge_test["recall"],    rf_test["recall"],    mlp_test["recall"]],
    "roc_auc":   [ridge_test["roc_auc"],   rf_test["roc_auc"],   mlp_test["roc_auc"]],
})

# -------------------------------------------------
# Plotting
# -------------------------------------------------
plt.figure(figsize=(12, 6))

x = np.arange(len(metrics))  # label positions
width = 0.25

plt.bar(x - width, data.iloc[0, 1:], width=width, label="Ridge")
plt.bar(x,         data.iloc[1, 1:], width=width, label="Random Forest")
plt.bar(x + width, data.iloc[2, 1:], width=width, label="MLP")

plt.xticks(x, metrics, fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Comparison of SMILES-Based Models on Test Set", fontsize=14)
plt.ylim(0, 1.1)

plt.legend()
plt.tight_layout()

plt.savefig("smiles_model_comparison.png", dpi=300)
plt.show()
