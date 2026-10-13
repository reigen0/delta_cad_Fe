#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import numpy as np
from cad import compute_cad

FEATURES = [
    "n_arg","density_ns","self_attack_ratio","reciprocal_attack_ratio_ns",
    "unattacked_ratio","no_outgoing_ratio","in_degree_cv_ns",
    "out_degree_cv_ns","largest_scc_ratio","cyclic_node_ratio",
    "sink_scc_node_ratio","condensation_depth_normalized",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--acceptance-dir", type=Path, required=True)
    p.add_argument("--n-select", type=int, default=700)
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()
#normalize features
def prepare_features(rows):
    x = {f: np.array([float(r[f]) for r in rows], dtype=float) for f in FEATURES}
    x["n_arg"] = np.log1p(x["n_arg"]) / np.log1p(x["n_arg"].max())
    x["in_degree_cv_ns"] = x["in_degree_cv_ns"] / (1 + x["in_degree_cv_ns"])
    x["out_degree_cv_ns"] = x["out_degree_cv_ns"] / (1 + x["out_degree_cv_ns"])
    return np.column_stack([x[f] for f in FEATURES])
#calc distances
def distance(points, reference):
    return np.mean(np.abs(points - reference), axis=1)
#select using farthest first
def farthest_first(points, n_select):
    if not 1 <= n_select <= len(points):
        raise ValueError("n-select must be between 1 and the number of eligible AFs")
    center = np.median(points, axis=0)
    first = int(np.argmin(distance(points, center)))
    selected = [first]
    nearest = distance(points, points[first])
    nearest[first] = -1.0

    while len(selected) < n_select:
        i = int(np.argmax(nearest))
        selected.append(i)
        nearest = np.minimum(nearest, distance(points, points[i]))
        nearest[i] = -1.0

    return selected

def main():
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [
        r for r in rows
        if r["eligible"] == "1"
    ]

    if not rows:
        raise ValueError("No eligible graphs in the input CSV")
    selected_idx = farthest_first(prepare_features(rows), args.n_select)
    selected = []
    reused = computed = failed = 0

    for rank, i in enumerate(selected_idx, start=1):
        row = dict(rows[i])
        row["selection_rank"] = rank
        cad_path = args.acceptance_dir / f"{row['file_name']}.dca.txt"

        accepted, failed_argument, was_computed = compute_cad(
            row["file_path"], args.solver,
            timeout=args.timeout, output_path=cad_path,
        )

        if accepted is None:
            row["cad_available"] = 0
            row["accepted_count"] = ""
            row["accepted_ratio"] = ""
            failed += 1
            print(f"CAD failed: {row['file_name']} at {failed_argument}")
        else:
            row["cad_available"] = 1
            row["accepted_count"] = len(accepted)
            row["accepted_ratio"] = len(accepted) / int(row["n_arg"])
            computed += int(was_computed)
            reused += int(not was_computed)

        selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["selection_rank"] + [k for k in selected[0] if k != "selection_rank"]
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(selected)

    print(
        f"prepool: {len(selected)} | "
        f"CAD reused: {reused} | computed: {computed} | failed: {failed}"
    )
    print(args.output)

if __name__ == "__main__":
    main()
