import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
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
# Constants / config
# -----------------------------
PICKLE_FILE = "plip_graphs.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Best hyperparameters from v2 sweep
HIDDEN_DIM = 32
NUM_LAYERS = 3
DROPOUT = 0.40
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 16
N_SPLITS = 5

# -----------------------------
# Load graphs
# -----------------------------
print("Loading graphs...")
with open(PICKLE_FILE, "rb") as f:
    graph_data = pickle.load(f)

graphs_raw = graph_data["graphs"]
print(f"Loaded {len(graphs_raw)} graphs.\n")

graphs = []
labels = []
names = []

for g in graphs_raw:
    data = Data(
        x=torch.tensor(g["x"], dtype=torch.float32),
        edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(g["edge_attr"], dtype=torch.float32),
        y=torch.tensor([float(g["y"])], dtype=torch.float32),
    )
    graphs.append(data)
    labels.append(int(g["y"]))
    names.append(g["name"])

labels = np.array(labels)

# -----------------------------
# Model definition
# -----------------------------
class PLIPGNN(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        # Separate MLPs for ligand vs protein nodes
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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # last feature dimension encodes "protein node" indicator
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
# Evaluation helper
# -----------------------------
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, loader):
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

    metrics = {}
    metrics["loss"] = criterion(torch.tensor(logits), torch.tensor(labels).float()).item()
    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["precision"] = precision_score(labels, preds, zero_division=0)
    metrics["recall"] = recall_score(labels, preds, zero_division=0)
    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["bal_acc"] = balanced_accuracy_score(labels, preds)
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["ppr"] = float(preds.mean())  # positive prediction rate

    metrics["logits"] = logits
    metrics["labels"] = labels
    metrics["probs"] = probs
    metrics["preds"] = preds

    return metrics


# -----------------------------
# Cross-validation loop
# -----------------------------
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=23,
)

all_fold_metrics = []
metric_names = ["loss", "accuracy", "precision", "recall", "f1", "bal_acc", "roc_auc", "ppr"]

in_dim = graphs[0].x.size(1)
edge_dim = graphs[0].edge_attr.size(1)

fold_id = 0

for train_idx, val_idx in skf.split(np.arange(len(graphs)), labels):
    fold_id += 1
    print("\n" + "=" * 37)
    print(f"Fold {fold_id}")
    print("=" * 37)

    # Optional: change seed slightly each fold for robustness
    set_seed(23 + fold_id)

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

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

    best_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses = []

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_metrics = evaluate(model, val_loader)
        val_auc = val_metrics["roc_auc"]

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_loss:.4f} | "
            f"AUC {val_auc:.3f}"
        )

        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best model and evaluate on this fold's val set
    if best_state is not None:
        model.load_state_dict(best_state)

    fold_metrics = evaluate(model, val_loader)

    print(f"Fold {fold_id} final AUC = {fold_metrics['roc_auc']:.3f}")
    print(f"Fold {fold_id} final F1  = {fold_metrics['f1']:.3f}")
    print(f"Fold {fold_id} final PPR = {fold_metrics['ppr']:.3f}")

    # Store only numeric metrics
    all_fold_metrics.append({m: fold_metrics[m] for m in metric_names})


# -----------------------------
# Aggregate CV metrics
# -----------------------------
print("\nWriting CV metrics to csv...")

import csv

out_file = "gnn_plip_cv_metrics_final.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["fold"] + metric_names
    writer.writerow(header)

    for i, fm in enumerate(all_fold_metrics, start=1):
        row = [i] + [fm[m] for m in metric_names]
        writer.writerow(row)

    # mean and std rows
    means = {m: float(np.mean([fm[m] for fm in all_fold_metrics])) for m in metric_names}
    stds  = {m: float(np.std([fm[m] for fm in all_fold_metrics], ddof=0)) for m in metric_names}

    writer.writerow(["mean"] + [means[m] for m in metric_names])
    writer.writerow(["std"]  + [stds[m] for m in metric_names])

print(f"\n=====================================")
print("Cross-validation complete!")
print("Summary:")

for m in metric_names:
    mu = np.mean([fm[m] for fm in all_fold_metrics])
    sd = np.std([fm[m] for fm in all_fold_metrics], ddof=0)
    print(f"{m:10s}: {mu:.3f} ± {sd:.3f}")

print("=====================================")
print(f"Saved CV metrics to {out_file}")