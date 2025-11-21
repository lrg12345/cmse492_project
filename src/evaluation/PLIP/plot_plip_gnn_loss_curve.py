import matplotlib.pyplot as plt

# Manually extracted from your training log:
train_losses = [
    3.1739, 1.7977, 1.5852, 1.3572, 1.0690,
    1.1116, 0.9048, 0.7485, 0.7063, 0.6607,
    0.6457, 0.6450, 0.6582, 0.6557, 0.6746,
    0.6741, 0.6732, 0.6637, 0.6707, 0.6736
]

val_auc = [
    0.610, 0.746, 0.466, 0.742, 0.773,
    0.583, 0.557, 0.648, 0.610, 0.659,
    0.655, 0.712, 0.705, 0.712, 0.689,
    0.701, 0.712, 0.655, 0.667, 0.689
]

epochs = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 5))

plt.plot(epochs, train_losses, marker="o", label="Train Loss")
plt.plot(epochs, val_auc, marker="s", label="Validation AUC", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss / AUC")
plt.title("PLIP-GNN Training Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("plip_gnn_training_curve.png", dpi=300)
plt.show()
