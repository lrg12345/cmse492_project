import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GraphNorm

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(23)

# -----------------------------
# File paths & constants
# -----------------------------
PICKLE_FILE = "plip_graphs.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best hyperparameters from sweep
HIDDEN_DIM = 32
NUM_LAYERS = 3
DROPOUT = 0.40
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16

# -----------------------------
# Load processed PLIP graphs
# -----------------------------
print(f"Loading graphs from {PICKLE_FILE} ...")
with open(PICKLE_FILE, "rb") as f:
    graph_data = pickle.load(f)

graphs_raw = graph_data["graphs"]
print(f"Total graphs: {len(graphs_raw)}")

# Convert to PyG Data objects
graphs = []
labels = []
names = []

for g in graphs_raw:
    data = Data(
        x=torch.tensor(g["x"], dtype=torch.float32),
        edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(g["edge_attr"], dtype=torch.float32),
        y=torch.tensor([float(g["y"])], dtype=torch.float32)
    )
    graphs.append(data)
    labels.append(int(g["y"]))
    names.append(g["name"])

labels = np.array(labels)
print(f"Positives: {labels.sum()} | Negatives: {len(labels) - labels.sum()}")

# -----------------------------
# Train/val/test split
# -----------------------------
idx_all = np.arange(len(graphs))

idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx_all, labels,
    test_size=0.30, stratify=labels, random_state=23
)

idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp,
    test_size=0.50, stratify=y_temp, random_state=23
)

def gs(idxs): return [graphs[i] for i in idxs]

train_graphs = gs(idx_train)
val_graphs = gs(idx_val)
test_graphs = gs(idx_test)

print("Split sizes:")
print(f"  Train: {len(train_graphs)}")
print(f"  Val:   {len(val_graphs)}")
print(f"  Test:  {len(test_graphs)}")

train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Model definition
# -----------------------------
class PLIPGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lig_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prot_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn_layer, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        is_prot = x[:, -1] > 0.5
        h_lig = self.lig_mlp(x)
        h_prot = self.prot_mlp(x)
        h = torch.where(is_prot.unsqueeze(1), h_prot, h_lig)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h, batch)
            h = F.relu(h)
            h = self.dropout(h)

        g = global_add_pool(h, batch)
        out = self.fc_out(g)
        return out.view(-1)

# -----------------------------
# Initialize model
# -----------------------------
in_dim = graphs[0].x.size(1)
edge_dim = graphs[0].edge_attr.size(1)

model = PLIPGNN(
    in_dim=in_dim,
    edge_dim=edge_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.BCEWithLogitsLoss()

# -----------------------------
# Evaluation helper
# -----------------------------
def evaluate(loader):
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits_list.append(logits.cpu().numpy())
            labels_list.append(batch.y.view(-1).cpu().numpy())

    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    out = {}
    out["loss"] = criterion(torch.tensor(logits), torch.tensor(labels).float()).item()
    out["accuracy"] = accuracy_score(labels, preds)
    out["precision"] = precision_score(labels, preds, zero_division=0)
    out["recall"] = recall_score(labels, preds, zero_division=0)
    out["f1"] = f1_score(labels, preds, zero_division=0)
    out["bal_acc"] = balanced_accuracy_score(labels, preds)
    try:
        out["roc_auc"] = roc_auc_score(labels, probs)
    except:
        out["roc_auc"] = float("nan")

    out["logits"] = logits
    out["labels"] = labels
    out["probs"] = probs
    out["preds"] = preds

    return out

# -----------------------------
# Training loop with early stopping (AUC-based)
# -----------------------------
best_auc = -1
best_state = None
epochs_no_improve = 0

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    losses = []

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    train_loss = np.mean(losses)

    val_metrics = evaluate(val_loader)
    val_auc = val_metrics["roc_auc"]

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Val AUC {val_auc:.3f}"
    )

    if val_auc > best_auc + 1e-4:
        best_auc = val_auc
        best_state = model.state_dict().copy()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered.\n")
            break

model.load_state_dict(best_state)

# -----------------------------
# Final evaluations
# -----------------------------
print("Final evaluation on train / val / test...\n")

train_m = evaluate(train_loader)
val_m = evaluate(val_loader)
test_m = evaluate(test_loader)

def print_block(name, m):
    print(f"{name}:")
    print(f"  loss      : {m['loss']:.4f}")
    print(f"  accuracy  : {m['accuracy']:.3f}")
    print(f"  precision : {m['precision']:.3f}")
    print(f"  recall    : {m['recall']:.3f}")
    print(f"  f1        : {m['f1']:.3f}")
    print(f"  bal_acc   : {m['bal_acc']:.3f}")
    print(f"  roc_auc   : {m['roc_auc']:.3f}\n")

print_block("Train", train_m)
print_block("Val", val_m)
print_block("Test", test_m)

# -----------------------------
# Save test predictions
# -----------------------------
import csv

with open("gnn_plip_test_predictions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["compound", "true_label", "prob_positive", "pred_label"])
    for name, lbl, prob, pred in zip(
        [names[i] for i in idx_test],
        test_m["labels"], test_m["probs"], test_m["preds"]
    ):
        writer.writerow([name, int(lbl), float(prob), int(pred)])

with open("gnn_plip_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "loss", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"])
    writer.writerow(["train", train_m["loss"], train_m["accuracy"], train_m["precision"],
                     train_m["recall"], train_m["f1"], train_m["bal_acc"], train_m["roc_auc"]])
    writer.writerow(["val", val_m["loss"], val_m["accuracy"], val_m["precision"],
                     val_m["recall"], val_m["f1"], val_m["bal_acc"], val_m["roc_auc"]])
    writer.writerow(["test", test_m["loss"], test_m["accuracy"], test_m["precision"],
                     test_m["recall"], test_m["f1"], test_m["bal_acc"], test_m["roc_auc"]])

print("Saved test predictions to gnn_plip_test_predictions.csv")
print("Saved metrics to gnn_plip_metrics.csv\n")
print("✅ Training complete.")