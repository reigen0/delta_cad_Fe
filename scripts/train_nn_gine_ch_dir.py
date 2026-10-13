import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GINEConv

from evaluation import best_mcc_threshold, compute_metrics, masked_scores, predict_labels
from prepare_pyg_data import load_split, make_loader

# define paths
DATASET_DIR = Path("dataset")
SMALL_DIR = Path("ds_small")
OUTPUT_DIR = Path("results_gine")
# set feature set
FEATURE_SET = "struct_cad"  # basic, basic_cad, struct, struct_cad
USE_EXHAUSTIVE = True  # True: use additional small dataset, else set False
USE_C_MASK = True  # if True: restrict TRAINING LOSS to C_delta only, else set False
# set hyperparameters, used params are:
AUX_WEIGHT = 0.1

BATCH_SIZE = 8
HIDDEN_DIM = 64
NUM_LAYERS = 3
DROPOUT = 0.1
LR = 0.001
WEIGHT_DECAY = 0.00001
MAX_EPOCHS = 120
PATIENCE = 14
SEED = 3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(SEED)

# define model
class GINEModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input = nn.Linear(input_dim, HIDDEN_DIM)
        self.convs = nn.ModuleList()

        for _ in range(NUM_LAYERS):
            mlp = nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            self.convs.append(GINEConv(mlp, edge_dim=1))
        self.norms = nn.ModuleList([
            nn.LayerNorm(HIDDEN_DIM) for _ in range(NUM_LAYERS)
        ])
        self.change_out = nn.Linear(HIDDEN_DIM, 1)
        self.direction_out = nn.Linear(HIDDEN_DIM, 2)

    def forward(self, data):
        x = self.input(data.x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, data.edge_index, data.edge_attr))
            if i < NUM_LAYERS - 1:
                x = F.dropout(x, p=DROPOUT, training=self.training)
        return self.change_out(x).squeeze(-1), self.direction_out(x)

# define training loop
def train_epoch(net, loader, optim, change_loss_fn, direction_loss_fn):
    net.train()
    change_sum = direction_sum = 0.0
    n_nodes = n_changed = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optim.zero_grad()
        change_logits, direction_logits = net(batch)
        y_change, y_dir = batch.y, batch.y_dir

        if USE_C_MASK:
            if not batch.c_mask.any():
                continue
            change_logits = change_logits[batch.c_mask]
            direction_logits = direction_logits[batch.c_mask]
            y_change = y_change[batch.c_mask]
            y_dir = y_dir[batch.c_mask]

        change_loss = change_loss_fn(change_logits, y_change)
        direction_active = y_dir >= 0

        if direction_active.any():
            direction_loss = direction_loss_fn(
                direction_logits[direction_active],
                y_dir[direction_active],
            )
            loss = change_loss + AUX_WEIGHT * direction_loss
            n = int(direction_active.sum())
            direction_sum += direction_loss.item() * n
            n_changed += n
        else:
            loss = change_loss

        loss.backward()
        optim.step()

        change_sum += change_loss.item() * y_change.numel()
        n_nodes += y_change.numel()

    return change_sum / n_nodes, direction_sum / n_changed if n_changed else 0.0

# define prediction loop
@torch.no_grad()
def predict(net, loader):
    net.eval()
    y, probs, masks = [], [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        change_logits, _ = net(batch)
        y.append(batch.y.cpu())
        probs.append(torch.sigmoid(change_logits).cpu())
        masks.append(batch.c_mask.cpu())

    return tuple(torch.cat(x).numpy() for x in (y, probs, masks))

# Save raw predictions and the C_delta-filtered prediction.
def save_predictions(dataset, probs, threshold, path):
    """Save raw predictions and the C_delta-filtered prediction."""
    rows, offset = [], 0

    for data in dataset:
        n = data.num_nodes
        p = probs[offset:offset + n]
        mask = data.c_mask.numpy()

        raw_pred = predict_labels(p, threshold)
        cfilter_scores = masked_scores(p, mask)
        cfilter_pred = predict_labels(p, threshold, mask)

        for i, argument in enumerate(data.node_names):
            rows.append({
                "graph_id": data.graph_id,
                "file_name": data.file_name,
                "source": data.source,
                "target": data.target,
                "argument": argument,
                "y_true": int(data.y[i]),
                "p_change_raw": float(p[i]),
                "p_change": float(p[i]),
                "p_change_cfilter": float(cfilter_scores[i]),
                "y_pred": int(raw_pred[i]),
                "y_pred_cfilter": int(cfilter_pred[i]),
                "in_c_delta": int(mask[i]),
            })
        offset += n

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    train = load_split(DATASET_DIR, "train", FEATURE_SET)
    if USE_EXHAUSTIVE:
        train += load_split(SMALL_DIR, "train", FEATURE_SET)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(DATASET_DIR, "val", FEATURE_SET, BATCH_SIZE)
    test_loader = make_loader(DATASET_DIR, "test", FEATURE_SET, BATCH_SIZE)

    net = GINEModel(train[0].x.size(1)).to(DEVICE)
    change_loss_fn = nn.BCEWithLogitsLoss()
    direction_loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_ap, best_epoch, wait = -1.0, 0, 0
    best_state = copy.deepcopy(net.state_dict())

    for epoch in range(1, MAX_EPOCHS + 1):
        change_loss, direction_loss = train_epoch(
            net, train_loader, optim, change_loss_fn, direction_loss_fn
        )
        y_val, p_val, val_mask = predict(net, val_loader)
        # Checkpoint selection is always based on unfiltered validation scores.
        # USE_C_MASK changes only the training loss.
        val_ap = average_precision_score(y_val, p_val)

        print(
            f"Epoch {epoch:03d} | change {change_loss:.4f} | "
            f"direction {direction_loss:.4f} | val AP {val_ap:.4f}"
        )

        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = copy.deepcopy(net.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break

    net.load_state_dict(best_state)

    y_val, p_val, val_mask = predict(net, val_loader)

    # Raw rule: threshold selected on unfiltered validation predictions.
    threshold, _ = best_mcc_threshold(y_val, p_val)
    val_metrics = compute_metrics(y_val, p_val, threshold)


    y_test, p_test, test_mask = predict(net, test_loader)
    test_metrics = compute_metrics(y_test, p_test, threshold)

    # Test-only C_delta prediction-filter analysis using the SAME threshold.
    test_cfilter_metrics = compute_metrics(
        y_test, p_test, threshold, test_mask
    )
    test_cfilter_fp_reduced = (
        test_metrics["fp"] - test_cfilter_metrics["fp"]
    )
# set output filename specifying the run
    run_name = (
        f"{FEATURE_SET}_auxdir_{'exh' if USE_EXHAUSTIVE else 'bench'}_"
        f"{'cmask' if USE_C_MASK else 'nomask'}_seed{SEED}"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(net.state_dict(), OUTPUT_DIR / f"{run_name}_model.pt")
    save_predictions(
        test_loader.dataset,
        p_test,
        threshold,
        OUTPUT_DIR / f"{run_name}_predictions.csv",
    )

    summary = {
        "feature_set": FEATURE_SET,
        "use_exhaustive": USE_EXHAUSTIVE,
        "use_c_loss_mask": USE_C_MASK,
        "aux_direction": True,
        "aux_weight": AUX_WEIGHT,
        "seed": SEED,
        "best_epoch": best_epoch,
        "threshold": threshold,
    }
    summary.update({f"val_{k}": v for k, v in val_metrics.items()})
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})
    summary.update({f"test_cfilter_{k}": v for k, v in test_cfilter_metrics.items()})
    summary["test_cfilter_fp_reduced"] = test_cfilter_fp_reduced


    with (OUTPUT_DIR / f"{run_name}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"best epoch: {best_epoch} | threshold: {threshold:.6f}")
    print(f"val raw AP/MCC: {val_metrics['ap']:.4f}/{val_metrics['mcc']:.4f}")
    print(f"test raw AP/MCC: {test_metrics['ap']:.4f}/{test_metrics['mcc']:.4f}")
    print(
        "test C-filter AP/MCC: "
        f"{test_cfilter_metrics['ap']:.4f}/{test_cfilter_metrics['mcc']:.4f}"
    )
    print(f"test C-filter FP reduced: {test_cfilter_fp_reduced}")



if __name__ == "__main__":
    main()
