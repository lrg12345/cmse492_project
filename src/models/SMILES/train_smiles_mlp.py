import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
print("Loading binary dataset...")

X = np.load("X_smiles_morgan2048.npy")
y = np.load("y_binary.npy")
compounds = np.load("compound_names.npy", allow_pickle=True)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Positives: {np.sum(y==1)}   Negatives: {np.sum(y==0)}")

# ------------------------------------------------------------
# Train/Val/Test split (STRATIFIED)
# ------------------------------------------------------------
X_train, X_temp, y_train, y_temp, compounds_train, compounds_temp = train_test_split(
    X, y, compounds, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test, compounds_val, compounds_test = train_test_split(
    X_temp, y_temp, compounds_temp,
    test_size=34, random_state=42, stratify=y_temp
)

print("\nDataset sizes:")
print(f" Train: {len(y_train)}")
print(f" Val:   {len(y_val)}")
print(f" Test:  {len(y_test)}")

# ------------------------------------------------------------
# OPTIONAL: Oversample minority class (training only)
# ------------------------------------------------------------
# from imblearn.over_sampling import RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# X_train, y_train = ros.fit_resample(X_train, y_train)
# print(f"After oversampling: Train size = {len(y_train)}")

# ------------------------------------------------------------
# Train MLP Classifier
# ------------------------------------------------------------
print("\nTraining MLP Classifier...")

mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256),
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=200,
    random_state=42
)

mlp.fit(X_train, y_train)

# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------
def evaluate(split_name, y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n{split_name} performance:")
    print(f" accuracy  : {acc:.4f}")
    print(f" f1        : {f1:.4f}")
    print(f" precision : {prec:.4f}")
    print(f" recall    : {rec:.4f}")
    print(f" roc_auc   : {auc:.4f}")

    return acc, f1, prec, rec, auc

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------
train_pred = mlp.predict(X_train)
train_proba = mlp.predict_proba(X_train)[:, 1]

val_pred = mlp.predict(X_val)
val_proba = mlp.predict_proba(X_val)[:, 1]

test_pred = mlp.predict(X_test)
test_proba = mlp.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# Print metrics
# ------------------------------------------------------------
train_metrics = evaluate("Train", y_train, train_pred, train_proba)
val_metrics = evaluate("Val", y_val, val_pred, val_proba)
test_metrics = evaluate("Test", y_test, test_pred, test_proba)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------
print("\nSaving test predictions to mlp_binary_test_predictions.csv ...")

df_test = pd.DataFrame({
    "compound": compounds_test,
    "true_label": y_test,
    "pred_label": test_pred,
    "pred_prob": test_proba
})
df_test.to_csv("mlp_binary_test_predictions.csv", index=False)

print("Saving metrics to mlp_binary_metrics.csv ...")

df_metrics = pd.DataFrame({
    "split": ["train", "val", "test"],
    "accuracy": [train_metrics[0], val_metrics[0], test_metrics[0]],
    "f1":        [train_metrics[1], val_metrics[1], test_metrics[1]],
    "precision": [train_metrics[2], val_metrics[2], test_metrics[2]],
    "recall":    [train_metrics[3], val_metrics[3], test_metrics[3]],
    "roc_auc":   [train_metrics[4], val_metrics[4], test_metrics[4]],
})
df_metrics.to_csv("mlp_binary_metrics.csv", index=False)

print("\n✔ MLP binary classification complete!")
