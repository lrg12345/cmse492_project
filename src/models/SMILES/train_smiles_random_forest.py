import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

# ----------------------------
# Load data
# ----------------------------
print("Loading binary dataset...")

X = np.load("X_smiles_morgan2048.npy")         # shape (225, 2048)
y = np.load("y_binary.npy")                    # shape (225,)
compounds = np.load("compound_names.npy", allow_pickle=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Positives: {np.sum(y==1)}   Negatives: {np.sum(y==0)}")

# ----------------------------
# Stratified split
# ----------------------------
X_temp, X_test, y_temp, y_test, compounds_temp, compounds_test = \
    train_test_split(
        X, y, compounds,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

X_train, X_val, y_train, y_val, compounds_train, compounds_val = \
    train_test_split(
        X_temp, y_temp, compounds_temp,
        test_size=0.20,
        random_state=42,
        stratify=y_temp
    )

print("\nDataset sizes:")
print(f" Train: {len(X_train)}")
print(f" Val:   {len(X_val)}")
print(f" Test:  {len(X_test)}")

# ----------------------------
# Train Random Forest
# ----------------------------
print("\nTraining Random Forest Classifier...")

clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",  # helps with the 78 vs 147 imbalance
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

# ----------------------------
# Evaluation function
# ----------------------------
def evaluate(split, Xs, ys):
    preds = clf.predict(Xs)
    proba = clf.predict_proba(Xs)[:, 1]  # needed for ROC-AUC

    metrics = {
        "accuracy": accuracy_score(ys, preds),
        "f1": f1_score(ys, preds),
        "precision": precision_score(ys, preds, zero_division=0),
        "recall": recall_score(ys, preds, zero_division=0),
        "roc_auc": roc_auc_score(ys, proba),
    }

    print(f"\n{split} performance:")
    for k, v in metrics.items():
        print(f" {k:10s}: {v:.4f}")

    return preds, metrics


train_preds, train_metrics = evaluate("Train", X_train, y_train)
val_preds,   val_metrics   = evaluate("Val",   X_val,   y_val)
test_preds,  test_metrics  = evaluate("Test",  X_test,  y_test)

# ----------------------------
# Save predictions
# ----------------------------
print("\nSaving test predictions to rf_binary_test_predictions.csv ...")

with open("rf_binary_test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["compound", "true_label", "predicted_label"])
    for name, true, pred in zip(compounds_test, y_test, test_preds):
        writer.writerow([name, int(true), int(pred)])

# ----------------------------
# Save metrics
# ----------------------------
print("Saving metrics to rf_binary_metrics.csv ...")

with open("rf_binary_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "accuracy", "f1", "precision", "recall", "roc_auc"])

    for split, m in [
        ("train", train_metrics),
        ("val",   val_metrics),
        ("test",  test_metrics)
    ]:
        writer.writerow([
            split,
            m["accuracy"],
            m["f1"],
            m["precision"],
            m["recall"],
            m["roc_auc"]
        ])

print("\n✔ Random Forest binary classification complete!\n")
