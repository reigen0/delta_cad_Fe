#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np

FEATURES = [
    "n_arg","density_ns","self_attack_ratio","reciprocal_attack_ratio_ns",
    "in_degree_cv_ns","out_degree_cv_ns","largest_scc_ratio",
    "cyclic_node_ratio","condensation_depth_normalized",
]
FEATURE_WEIGHTS = np.array([1,1,1,1,0.5,0.5,1,1,1], dtype=float)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-select", type=int, default=180)
    p.add_argument("--low-acceptance-threshold", type=float, default=0.2)
    p.add_argument("--max-low-acceptance-fraction", type=float, default=0.2)
    return p.parse_args()

# normalize features
def prepare_features(rows):
    x = {f: np.array([float(r[f]) for r in rows], dtype=float) for f in FEATURES}
    x["n_arg"] = np.log1p(x["n_arg"]) / np.log1p(x["n_arg"].max())
    x["in_degree_cv_ns"] = x["in_degree_cv_ns"] / (1 + x["in_degree_cv_ns"])
    x["out_degree_cv_ns"] = x["out_degree_cv_ns"] / (1 + x["out_degree_cv_ns"])
    return np.column_stack([x[f] for f in FEATURES])
#calc distances
def distance(points, reference):
    return np.average(np.abs(points - reference), axis=1, weights=FEATURE_WEIGHTS)
# select using farthest first
def farthest_first(points, accepted_ratio, n_select, threshold, max_low):
    low = accepted_ratio < threshold
    if np.count_nonzero(~low) + max_low < n_select:
        raise ValueError("Low-acceptance constraint makes selection infeasible.")

    center = np.median(points, axis=0)
    center_distance = distance(points, center)
    if max_low == 0:
        center_distance[low] = np.inf

    first = int(np.argmin(center_distance))
    selected = [first]
    n_low = int(low[first])
    nearest = distance(points, points[first])
    nearest[first] = -1.0

    while len(selected) < n_select:
        scores = nearest.copy()
        if n_low >= max_low:
            scores[low] = -1.0
        i = int(np.argmax(scores))
        if scores[i] < 0:
            raise ValueError("No eligible AF remains.")
        selected.append(i)
        n_low += int(low[i])
        nearest = np.minimum(nearest, distance(points, points[i]))
        nearest[i] = -1.0

    return selected

def main():
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["cad_available"] == "1" and r["accepted_ratio"] != ""]
    points = prepare_features(rows)
    accepted_ratio = np.array([float(r["accepted_ratio"]) for r in rows])
    max_low = int(args.n_select * args.max_low_acceptance_fraction)

    selected_idx = farthest_first(
        points, accepted_ratio, args.n_select,
        args.low_acceptance_threshold, max_low,
    )

    selected = []
    for rank, i in enumerate(selected_idx, start=1):
        row = dict(rows[i])
        row["final_selection_rank"] = rank
        selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["final_selection_rank"] + [k for k in selected[0] if k != "final_selection_rank"]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)

    n_low = sum(float(r["accepted_ratio"]) < args.low_acceptance_threshold for r in selected)
    print(f"final pool: {len(selected)} | low accepted-ratio: {n_low}/{max_low}")
    print(args.output)

if __name__ == "__main__":
    main()
