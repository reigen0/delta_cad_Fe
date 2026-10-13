#!/usr/bin/env python3
"""Enumerate and add every missing attack and solve resulting graph"""

import argparse
import csv
import tempfile
import time
from pathlib import Path

from cad import compute_cad, join_args, read_cad
from parse_graphs import read_graph


UPDATE_COLUMNS = [
    "graph_id", "file_name", "source", "target", "status", "runtime_seconds",
    "accepted_count_before", "accepted_count_after", "accepted_after_args",
    "gain_count", "loss_count", "delta_count",
    "gain_args", "loss_args", "delta_args", "outcome_type",
]
INSTANCE_COLUMNS = [
    "graph_id", "file_name", "source", "target",
    "delta_args", "gain_args", "loss_args",
]


# remark: just tgf and apx files used, no af files
def write_graph(graph, path):
    with path.open("w", encoding="utf-8") as f:
        if path.suffix.lower() == ".tgf":
            for node in graph:
                f.write(f"{node}\n")
            f.write("#\n")
            for a, b in graph.edges():
                f.write(f"{a} {b}\n")
        else:
            for node in graph:
                f.write(f"arg({node}).\n")
            for a, b in graph.edges():
                f.write(f"att({a},{b}).\n")


def outcome(gain, loss):
    if gain and loss:
        return "mixed"
    if gain:
        return "gain_only"
    if loss:
        return "loss_only"
    return "no_change"


def completed_updates(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            (r["graph_id"], r["source"], r["target"])
            for r in csv.DictReader(f)
            if r["status"] == "completed"
        }


def append_update(path, row):
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UPDATE_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)

# write instance files used for model input
def write_instances(updates_path, instances_path):
    with updates_path.open("r", encoding="utf-8", newline="") as src, \
         instances_path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=INSTANCE_COLUMNS)
        writer.writeheader()
        for row in reader:
            if row["status"] == "completed":
                writer.writerow({c: row[c] for c in INSTANCE_COLUMNS})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--base-cad-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--solver", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=60)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    updates_path = args.output_dir / "updates.csv"
    instances_path = args.output_dir / "instances.csv"
    done = completed_updates(updates_path)

    with args.pool.open("r", encoding="utf-8", newline="") as f:
        pool = [
            row for row in csv.DictReader(f)
            if row.get("eligible", "1") == "1"
        ]

    with tempfile.TemporaryDirectory(prefix="adm_exhaustive_") as tmp:
        tmp = Path(tmp)

        for number, row in enumerate(pool, 1):
            graph_id = row["graph_id"]
            file_name = row["file_name"]
            graph_path = Path(row["file_path"])
            if not graph_path.is_absolute():
                graph_path = args.root / graph_path

            graph = read_graph(graph_path)
            nodes = list(graph)
            before = read_cad(args.base_cad_dir / f"{file_name}.dca.txt")
            temp_graph = tmp / f"updated{graph_path.suffix.lower()}"

            for source in nodes:
                for target in nodes:
                    if graph.has_edge(source, target):
                        continue

                    key = (graph_id, source, target)
                    if key in done:
                        continue

                    graph.add_edge(source, target)
                    write_graph(graph, temp_graph)
                    graph.remove_edge(source, target)

                    start = time.perf_counter()
                    after, failed, _ = compute_cad(
                        temp_graph, args.solver, args.timeout
                    )
                    runtime = round(time.perf_counter() - start, 6)

                    if after is None:
                        print(
                            f"timeout: {file_name} + ({source},{target}), "
                            f"argument {failed}"
                        )
                        continue

                    gain = after - before
                    loss = before - after
                    delta = gain | loss

                    append_update(updates_path, {
                        "graph_id": graph_id,
                        "file_name": file_name,
                        "source": source,
                        "target": target,
                        "status": "completed",
                        "runtime_seconds": runtime,
                        "accepted_count_before": len(before),
                        "accepted_count_after": len(after),
                        "accepted_after_args": join_args(after),
                        "gain_count": len(gain),
                        "loss_count": len(loss),
                        "delta_count": len(delta),
                        "gain_args": join_args(gain),
                        "loss_args": join_args(loss),
                        "delta_args": join_args(delta),
                        "outcome_type": outcome(gain, loss),
                    })
                    done.add(key)

            print(f"[{number}/{len(pool)}] {file_name}")

    write_instances(updates_path, instances_path)
    print(f"Completed updates: {len(done)}")
    print(updates_path)
    print(instances_path)


if __name__ == "__main__":
    main()
