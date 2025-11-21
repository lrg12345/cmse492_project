#!/usr/bin/env python3
"""
PLIP-GNN hyperparameter sweep (V2)

- Uses the "new" PLIPGNN architecture:
  * separate ligand / protein MLPs
  * GINEConv with edge_attr
  * GraphNorm + Dropout

- Reuses a single stratified train/val/test split for ALL configs
- Evaluates:
    loss, accuracy, precision, recall, f1, balanced_accuracy, roc_auc, ppr
- Rejects configs that collapse (predict almost all negatives or positives)
- Selects best config primarily by F1 (with AUC/bal_acc helping)

Outputs:
  gnn_hyperparameter_sweep_v2_results.csv
"""

import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, GraphNorm


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(23)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_PICKLE = "plip_graphs.pkl"


# -----------------------------
# Load graphs
# -----------------------------
print(f"Loading graphs from {GRAPH_PICKLE}...")
with open(GRAPH_PICKLE, "rb") as f:
    data_dict = pickle.load(f)

graphs_raw = data_dict["graphs"]
feature_dim = data_dict.get("feature_dim", None)
atom_dim = data_dict.get("atom_dim", 21)
res_dim = data_dict.get("res_dim", 29)

graphs = []
labels = []
names = []

for g in graphs_raw:
    x = torch.tensor(g["x"], dtype=torch.float32)
    edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(g["edge_attr"], dtype=torch.float32)
    y = float(g["y"])
    name = g.get("name", "")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float32),
    )
    graphs.append(data)
    labels.append(int(y))
    names.append(name)

labels = np.array(labels, dtype=int)
print(f"Loaded {len(graphs)} graphs.")
print(f"Positives: {labels.sum()} | Negatives: {len(labels) - labels.sum()}")

if feature_dim is None:
    feature_dim = graphs[0].x.size(1)
print(f"Feature dim = {feature_dim}, atom_dim = {atom_dim}, res_dim = {res_dim}")

edge_dim = graphs[0].edge_attr.size(1)
print(f"Edge feature dim = {edge_dim}")


# -----------------------------
# Fixed stratified train/val/test split
# -----------------------------
idx_all = np.arange(len(graphs))

# 70% train, 15% val, 15% test
idx_train, idx_temp, y_train, y_temp = train_test_split(
    idx_all,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=23,
)

idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=23,
)


def subset_graphs(indices):
    return [graphs[i] for i in indices]


train_graphs = subset_graphs(idx_train)
val_graphs = subset_graphs(idx_val)
test_graphs = subset_graphs(idx_test)

print("\nSplit sizes:")
print(f"  Train: {len(train_graphs)}")
print(f"  Val  : {len(val_graphs)}")
print(f"  Test : {len(test_graphs)}")

# DataLoaders will be rebuilt per run (to reshuffle), but splits are fixed.


# -----------------------------
# Model definition (new PLIPGNN)
# -----------------------------
class PLIPGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        atom_dim: int = 21,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.atom_dim = atom_dim

        # Separate encoders for ligand atoms and protein residues
        self.lig_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prot_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GINEConv layers using edge_attr
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            nn_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            conv = GINEConv(nn=nn_layer, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))

        # Graph-level classifier
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Determine node types from feature layout:
        # atoms: first atom_dim entries non-zero, residue part ~0
        # residues: residue segment non-zero (we don't need exact res_dim here)
        residue_part = x[:, self.atom_dim:]
        res_mask = (residue_part.abs().sum(dim=1) > 1e-6)
        lig_mask = ~res_mask

        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        if lig_mask.any():
            h[lig_mask] = self.lig_mlp(x[lig_mask])
        if res_mask.any():
            h[res_mask] = self.prot_mlp(x[res_mask])

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_attr)
            h = norm(h, batch)
            h = F.relu(h)

        g = global_add_pool(h, batch)
        out = self.fc_out(g)  # [batch_size, 1]
        return out.view(-1)


# -----------------------------
# Evaluation utilities
# -----------------------------
criterion = nn.BCEWithLogitsLoss()


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch.y.view(-1).cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    # Basic metrics
    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, preds)

    # AUC (guard against single-class edge cases)
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = float("nan")

    logits_t = torch.from_numpy(all_logits)
    labels_t = torch.from_numpy(all_labels).float()
    loss = criterion(logits_t, labels_t).item()

    ppr = float((preds == 1).mean())  # positive prediction rate

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "bal_acc": bal_acc,
        "roc_auc": auc,
        "ppr": ppr,
        "logits": all_logits,
        "labels": all_labels,
        "probs": probs,
        "preds": preds,
    }


def score_config(val_metrics, ppr_min=0.05, ppr_max=0.95):
    """
    Composite score for hyperparameter selection.

    - Rejects configs that predict almost all negatives or positives
      by returning a large negative score.
    - Rewards F1 primarily, with bal_acc and AUC as helpers.
    """
    ppr = val_metrics["ppr"]

    if (ppr < ppr_min) or (ppr > ppr_max):
        # Collapsed model (degenerate classifier)
        return -1e9

    f1 = val_metrics["f1"]
    bal_acc = val_metrics["bal_acc"]
    auc = val_metrics["roc_auc"]
    if np.isnan(auc):
        auc = 0.5  # neutral

    # Simple linear combination:
    # - F1 is primary
    # - balanced accuracy helps
    # - AUC (above 0.5) gives a small bonus
    score = f1 + 0.3 * bal_acc + 0.2 * (auc - 0.5)
    return score


# -----------------------------
# Training for a single config
# -----------------------------
def train_one_config(
    hidden_dim,
    num_layers,
    dropout,
    lr,
    weight_decay,
    max_epochs=100,
    patience=20,
):
    # Rebuild loaders (to reshuffle train each config, but splits fixed)
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    model = PLIPGNN(
        in_channels=feature_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        atom_dim=atom_dim,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_score = -1e9
    best_state = None
    best_val_metrics = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        n_graphs = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_graphs += batch.num_graphs

        train_loss = total_loss / max(n_graphs, 1)

        val_metrics = evaluate(val_loader, model, DEVICE)
        val_score = score_config(val_metrics)

        print(
            f"  Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val AUC {val_metrics['roc_auc']:.3f} | "
            f"Val F1 {val_metrics['f1']:.3f} | "
            f"Val PPR {val_metrics['ppr']:.3f}"
        )

        if val_score > best_score + 1e-4:
            best_score = val_score
            best_state = model.state_dict()
            best_val_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    train_metrics = evaluate(train_loader, model, DEVICE)
    val_metrics = evaluate(val_loader, model, DEVICE)
    test_metrics = evaluate(test_loader, model, DEVICE)

    return train_metrics, val_metrics, test_metrics, best_score


# -----------------------------
# Hyperparameter grid (108 configs)
# -----------------------------
hidden_dims = [32, 64, 128]          # 3
num_layers_list = [2, 3, 4]          # 3
dropouts = [0.0, 0.2, 0.4]           # 3
lrs = [1e-4, 3e-4, 1e-3, 3e-3]       # 4
weight_decays = [1e-4]               # 1  --> 3*3*3*4*1 = 108 configs

configs = []
for H in hidden_dims:
    for L in num_layers_list:
        for dr in dropouts:
            for lr in lrs:
                for wd in weight_decays:
                    configs.append((H, L, dr, lr, wd))

print(f"\nTotal hyperparameter configs: {len(configs)}\n")

results = []
best_overall = {
    "score": -1e9,
    "config": None,
    "val_metrics": None,
    "test_metrics": None,
}

# -----------------------------
# Main sweep loop
# -----------------------------
for idx, (H, L, dr, lr, wd) in enumerate(configs, start=1):
    print("=" * 60)
    print(f"Config {idx}/{len(configs)}: H={H}, L={L}, DR={dr}, LR={lr}, WD={wd}")
    print("=" * 60)

    set_seed(23)  # reset seed for fair comparison
    train_m, val_m, test_m, score = train_one_config(
        hidden_dim=H,
        num_layers=L,
        dropout=dr,
        lr=lr,
        weight_decay=wd,
        max_epochs=100,
        patience=20,
    )

    # Record results
    for split_name, m in [("train", train_m), ("val", val_m), ("test", test_m)]:
        results.append(
            {
                "hidden_dim": H,
                "num_layers": L,
                "dropout": dr,
                "lr": lr,
                "weight_decay": wd,
                "split": split_name,
                "score": score if split_name == "val" else np.nan,
                "loss": m["loss"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "bal_acc": m["bal_acc"],
                "roc_auc": m["roc_auc"],
                "ppr": m["ppr"],
            }
        )

    # Track best config (based on VAL score)
    if score > best_overall["score"]:
        best_overall["score"] = score
        best_overall["config"] = (H, L, dr, lr, wd)
        best_overall["val_metrics"] = val_m
        best_overall["test_metrics"] = test_m
        print("\n*** New best config found! ***")
        print(f"  Config: H={H}, L={L}, DR={dr}, LR={lr}, WD={wd}")
        print(f"  Val F1={val_m['f1']:.3f}, AUC={val_m['roc_auc']:.3f}, PPR={val_m['ppr']:.3f}\n")


# -----------------------------
# Save results and print best config
# -----------------------------
df = pd.DataFrame(results)
out_csv = "gnn_hyperparameter_sweep_v2_results.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved hyperparameter sweep results to {out_csv}\n")

best_cfg = best_overall["config"]
best_val = best_overall["val_metrics"]
best_test = best_overall["test_metrics"]

print("=== Best config based on VAL composite score ===")
print(f"  H={best_cfg[0]}, L={best_cfg[1]}, DR={best_cfg[2]}, LR={best_cfg[3]}, WD={best_cfg[4]}")
print("  VAL:")
print(f"    F1       : {best_val['f1']:.3f}")
print(f"    Precision: {best_val['precision']:.3f}")
print(f"    Recall   : {best_val['recall']:.3f}")
print(f"    Bal Acc  : {best_val['bal_acc']:.3f}")
print(f"    AUC      : {best_val['roc_auc']:.3f}")
print(f"    PPR      : {best_val['ppr']:.3f}")
print("  TEST:")
print(f"    F1       : {best_test['f1']:.3f}")
print(f"    Precision: {best_test['precision']:.3f}")
print(f"    Recall   : {best_test['recall']:.3f}")
print(f"    Bal Acc  : {best_test['bal_acc']:.3f}")
print(f"    AUC      : {best_test['roc_auc']:.3f}")
print(f"    PPR      : {best_test['ppr']:.3f}")
print("===============================================")