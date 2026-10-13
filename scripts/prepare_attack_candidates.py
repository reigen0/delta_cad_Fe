#!/usr/bin/env python3
"""Describe all missing attacks from one eligible graph pool"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from parse_graphs import read_graph
from cad import read_cad
from attack_properties import graph_cache, describe_attack


COLUMNS = ["source", "target", "theory_class", "direction_class", "sampling_class",
           "source_acc_F", "target_acc_F", "self_attack_addition", "reciprocal",
           "gain_ruled_out", "loss_ruled_out", "change_guaranteed", "change_ruled_out",
           "cycle_closing", "target_reach_count", "change_C_cad_size"]
SUMMARY_COLUMNS = ["graph_id", "file_name", "candidate_file", "n_arg", "n_candidates", "n_self",
                   "n_guaranteed", "n_ruled_out", "n_unresolved",
                   "n_unresolved_gain_ruled_out", "n_unresolved_loss_ruled_out",
                   "n_unresolved_neither_ruled_out"]


def write_candidates(graph, accepted, output_path):
    cache = graph_cache(graph, accepted)
    existing, counts = set(graph.edges()), Counter()
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for a in cache["ordered_nodes"]:
            for b in cache["ordered_nodes"]:
                if (a, b) in existing:
                    continue
                fields, _, _ = describe_attack(a, b, cache)
                writer.writerow({c: int(fields[c]) if isinstance(fields[c], bool) else fields[c]
                                 for c in COLUMNS})
                counts["candidates"] += 1
                counts["self"] += a == b
                counts[fields["theory_class"]] += 1
                if fields["theory_class"] == "unresolved":
                    counts[fields["sampling_class"]] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("."), help="Resolve relative file_path values")
    p.add_argument("--acceptance-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    with args.pool.open(encoding="utf-8", newline="") as f:
        pool = list(csv.DictReader(f))
    by_graph = args.output_dir / "by_graph"
    by_graph.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, row in enumerate(pool, 1):
        path = Path(row["file_path"])
        graph = read_graph(path if path.is_absolute() else args.root / path)
        accepted = read_cad(args.acceptance_dir / f"{row['file_name']}.dca.txt")
        candidate_file = f"{row['graph_id']}__{row['file_name']}.candidates.csv"
        counts = write_candidates(graph, accepted, by_graph / candidate_file)
        record = dict(graph_id=row["graph_id"], file_name=row["file_name"],
                      candidate_file=candidate_file, n_arg=len(graph))
        record.update({c: counts[c[2:]] for c in SUMMARY_COLUMNS[4:]})
        summary.append(record)
        print(f"[{i}/{len(pool)}] {row['file_name']}: {counts['candidates']:,} candidates")
    with (args.output_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
