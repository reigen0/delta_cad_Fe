#!/usr/bin/env python3
"""Merge explicitly supplied pools; optionally read/compute their base acceptance."""
import argparse
import csv
from pathlib import Path


# read a single graph pool
def read_pool(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# merge graph pools into a single benchmark pool
def merge_pools(paths):
    rows, seen, duplicates = [], set(), 0
    for path in paths:
        for row in read_pool(path):
            if row.get("eligible", "1") != "1":
                continue
            # Same labeled-graph hash as the scanner; the first input wins.
            digest = row["graph_hash"]
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            row.pop("pool_group", None)  # Ignore obsolete metadata, not graph source.
            rows.append(row)
    if not rows:
        raise ValueError("No eligible graphs in the supplied pools")
    return rows, duplicates


# merge pools and compute CAD availability
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--acceptance-dir", type=Path)
    p.add_argument("--solver", type=Path)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()
    rows, duplicates = merge_pools(args.pools)
    if len({r["file_name"] for r in rows}) != len(rows):
        raise ValueError("Different AFs share a file_name; resolve this before copying/CAD reuse")
    if args.acceptance_dir is not None:
        from cad import compute_cad
        for row in rows:
            path = args.acceptance_dir / f"{row['file_name']}.dca.txt"
            if not path.exists() and args.solver is None:
                raise ValueError(f"CAD missing for {row['file_name']}; supply --solver")
            accepted, failed_argument, _ = compute_cad(
                row["file_path"], args.solver, timeout=args.timeout, output_path=path)
            row["cad_available"] = int(accepted is not None)
            row["accepted_count"] = len(accepted) if accepted is not None else ""
            row["accepted_ratio"] = len(accepted) / int(row["n_arg"]) if accepted is not None else ""
            if accepted is None:
                print(f"CAD pending: {row['file_name']} at {failed_argument}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"benchmark: {len(rows)} | duplicates removed: {duplicates}")
    if args.acceptance_dir is not None:
        print(f"CAD available: {sum(int(r['cad_available']) for r in rows)}/{len(rows)}")
    print(args.output)


if __name__ == "__main__":
    main()
