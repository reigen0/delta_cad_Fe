import csv
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from evaluation import best_mcc_threshold, compute_metrics, predict_labels
from parse_graphs import read_graph


DATASET_DIR = Path("dataset")
SMALL_DIR = Path("ds_small")
OUTPUT_DIR = Path("results_rf")

USE_CAD = True # false: use only structural features
USE_EXHAUSTIVE = True # true: use additional small dataset, else set False
# set hyperparameters
N_ESTIMATORS = 100
SEED = 3


def read_args(text):
    return {x.strip() for x in (text or "").replace("\n", ",").split(",") if x.strip()}

# cache graph data for faster computation
def graph_cache(G):
    nodes = list(G)
    denom = max(1, len(nodes) - 1)
    scc_of = {
        node: component
        for component in nx.strongly_connected_components(G)
        for node in component
    }
    degree = {}
    for node in nodes:
        self_attack = int(G.has_edge(node, node))
        degree[node] = (
            (G.in_degree(node) - self_attack) / denom,
            (G.out_degree(node) - self_attack) / denom,
        )
    return nodes, denom, scc_of, degree, {}

# compute features for a single pair of nodes
def make_features(G, source, target, accepted, cache):
    nodes, denom, scc_of, degree, distances = cache
    a_in = set(G.predecessors(source))
    a_out = set(G.successors(source))
    b_in = set(G.predecessors(target))
    b_out = set(G.successors(target))

    if target not in distances:
        distances[target] = nx.single_source_shortest_path_length(G, target)
    distance = distances[target]
    source_in, source_out = degree[source]
    target_in, target_out = degree[target]

    rows = []
    for node in nodes:
        node_in, node_out = degree[node]
        d = distance.get(node)
        row = [
            node == source,
            node == target,
            node in a_in,
            node in a_out,
            node in b_in,
            node in b_out,
            node in scc_of[target],
            node_in,
            node_out,
            source_in,
            source_out,
            target_in,
            target_out,
            G.has_edge(node, node),
            G.has_edge(source, source),
            G.has_edge(target, target),
            source == target,
            G.has_edge(target, source),
            -1.0 if d is None else d / denom,
        ]
        if accepted is not None:
            row.append(node in accepted)
        rows.append(row)

    return np.asarray(rows, dtype=np.float32)

# load a split of the dataset
def load_split(dataset_dir, split, use_cad, keep_rows=False):
    split_dir = Path(dataset_dir) / split
    instances = pd.read_csv(
        split_dir / "instances.csv",
        dtype=str,
        keep_default_na=False,
    )

    x_all, y_all = [], []
    records = [] if keep_rows else None

    for file_name, part in instances.groupby("file_name", sort=False):
        G = read_graph(split_dir / "base_graphs" / file_name)
        cache = graph_cache(G)
        nodes = cache[0]
        accepted = None

        if use_cad:
            accepted = read_args(
                (split_dir / "base_cad" / f"{file_name}.dca.txt").read_text(encoding="utf-8")
            )

        for row in part.itertuples(index=False):
            x = make_features(G, row.source, row.target, accepted, cache)
            delta = read_args(row.delta_args)
            y = np.asarray([node in delta for node in nodes], dtype=np.int8)
            x_all.append(x)
            y_all.append(y)

            if keep_rows:
                records.extend({
                    "graph_id": row.graph_id,
                    "file_name": file_name,
                    "source": row.source,
                    "target": row.target,
                    "argument": node,
                    "y_true": int(label),
                } for node, label in zip(nodes, y))

    return (
        np.concatenate(x_all),
        np.concatenate(y_all),
        pd.DataFrame(records) if keep_rows else None,
    )


def main():
    x_train, y_train, _ = load_split(DATASET_DIR, "train", USE_CAD)
    x_val, y_val, _ = load_split(DATASET_DIR, "val", USE_CAD)
    x_test, y_test, test_rows = load_split(DATASET_DIR, "test", USE_CAD, keep_rows=True)

    if USE_EXHAUSTIVE:
        x_small, y_small, _ = load_split(SMALL_DIR, "train", USE_CAD)
        x_train = np.concatenate([x_train, x_small])
        y_train = np.concatenate([y_train, y_small])

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    positive = int(np.flatnonzero(model.classes_ == 1)[0])
    p_val = model.predict_proba(x_val)[:, positive]
    threshold, _ = best_mcc_threshold(y_val, p_val)
    val_metrics = compute_metrics(y_val, p_val, threshold)

    p_test = model.predict_proba(x_test)[:, positive]
    test_metrics = compute_metrics(y_test, p_test, threshold)
    test_rows["p_change"] = p_test
    test_rows["y_pred"] = predict_labels(p_test, threshold).astype(int)

    feature_name = "rf_struct+cad" if USE_CAD else "rf_struct"
    run_name = f"{feature_name}_{'exh' if USE_EXHAUSTIVE else 'bench'}_{N_ESTIMATORS}_seed{SEED}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, OUTPUT_DIR / f"{run_name}_model.joblib")
    test_rows.to_csv(OUTPUT_DIR / f"{run_name}_predictions.csv", index=False)

    summary = {
        "feature_set": feature_name,
        "use_exhaustive": USE_EXHAUSTIVE,
        "seed": SEED,
        "n_estimators": N_ESTIMATORS,
        "threshold": threshold,
    }
    summary.update({f"val_{k}": v for k, v in val_metrics.items()})
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    with (OUTPUT_DIR / f"{run_name}_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print(f"threshold: {threshold:.6f}")
    print(f"val AP/MCC: {val_metrics['ap']:.4f}/{val_metrics['mcc']:.4f}")
    print(f"test AP/MCC: {test_metrics['ap']:.4f}/{test_metrics['mcc']:.4f}")


if __name__ == "__main__":
    main()
