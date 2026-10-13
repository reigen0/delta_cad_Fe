#!/usr/bin/env python3
"""Scan .tgf and .apx graphs in one directory. Keep all graph records, mark duplicates and exclusions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import networkx as nx
import numpy as np
from parse_graphs import read_graph


def graph_hash(graph):
    canonical = [
        sorted(str(node) for node in graph.nodes),
        sorted(
            (str(source), str(target))
            for source, target in graph.edges
        ),
    ]
    encoded = json.dumps(
        canonical,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def degree_cv(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean() if len(values) else 0.0
    return 0.0 if mean == 0 else float(values.std() / mean)


def graph_properties(graph):
    nodes = list(graph.nodes())
    n = len(nodes)
    n_att = graph.number_of_edges()
    self_edges = [(x, x) for x in nodes if graph.has_edge(x, x)]
    n_self = len(self_edges)

    nonself = graph.copy()
    nonself.remove_edges_from(self_edges)
    n_att_ns = nonself.number_of_edges()

    in_full = np.array([graph.in_degree(x) for x in nodes], dtype=float)
    out_full = np.array([graph.out_degree(x) for x in nodes], dtype=float)
    in_ns = np.array([nonself.in_degree(x) for x in nodes], dtype=float)
    out_ns = np.array([nonself.out_degree(x) for x in nodes], dtype=float)

    reciprocal = sum(1 for a, b in nonself.edges() if nonself.has_edge(b, a))
    sccs = list(nx.strongly_connected_components(nonself))
    n_scc = len(sccs)
    largest_scc = max((len(c) for c in sccs), default=0)

    cyclic = {x for comp in sccs if len(comp) > 1 for x in comp}
    cyclic.update(x for x in nodes if graph.has_edge(x, x))

    condensation = nx.condensation(nonself, sccs)
    sink_scc_nodes = sum(
        len(sccs[i]) for i in condensation.nodes()
        if condensation.out_degree(i) == 0
    )

    weak = list(nx.weakly_connected_components(nonself))
    largest_weak = max((len(c) for c in weak), default=0)

    cond_depth = 0.0 if n_scc <= 1 else (
            nx.dag_longest_path_length(condensation) / (n_scc - 1)
    )

    if n <= 1:
        reachable_pair_ratio = 0.0
    else:
        reachable_pairs = 0
        for i, comp in enumerate(sccs):
            size_i = len(comp)
            reachable_pairs += size_i * (size_i - 1)
            for j in nx.descendants(condensation, i):
                reachable_pairs += size_i * len(sccs[j])
        reachable_pair_ratio = reachable_pairs / (n * (n - 1))

    return {
        "n_arg": n,
        "n_att": n_att,
        "n_att_ns": n_att_ns,
        "non_att": n * n - n_att,
        "density_ns": n_att_ns / (n * (n - 1)) if n > 1 else 0.0,
        "self_attack_ratio": n_self / n if n else 0.0,
        "reciprocal_attack_ratio_ns": reciprocal / n_att_ns if n_att_ns else 0.0,
        "unattacked_ratio": float((in_full == 0).mean()) if n else 0.0,
        "no_outgoing_ratio": float((out_full == 0).mean()) if n else 0.0,
        "in_degree_cv_ns": degree_cv(in_ns),
        "out_degree_cv_ns": degree_cv(out_ns),
        "n_scc": n_scc,
        "largest_scc_ratio": largest_scc / n if n else 0.0,
        "cyclic_node_ratio": len(cyclic) / n if n else 0.0,
        "sink_scc_node_ratio": sink_scc_nodes / n if n else 0.0,
        "n_weak_components": len(weak),
        "largest_weak_component_ratio": largest_weak / n if n else 0.0,
        "condensation_depth_normalized": cond_depth,
        "reachable_pair_ratio": reachable_pair_ratio,
    }


def scan_graphs(input_dir, max_args=700, max_attacks=30000):
    input_dir = Path(input_dir).resolve()
    # AppleDouble/resource-fork files are metadata, not argumentation frameworks.
    paths = sorted(p for p in input_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".tgf", ".apx"}
                   and not p.name.startswith("._") and "__MACOSX" not in p.parts)
    rows, first_path = [], {}
    for path in paths:
        graph = read_graph(path)
        digest = graph_hash(graph)
        props = graph_properties(graph)
        duplicate_of = first_path.get(digest, "")
        first_path.setdefault(digest, str(path))
        reasons = []
        if props["n_arg"] > max_args:
            reasons.append("max_args")
        if props["n_att"] > max_attacks:
            reasons.append("max_attacks")
        if duplicate_of:
            reasons.append("duplicate")
        relative = path.relative_to(input_dir)
        rows.append({
            "graph_id": f"g_{digest[:16]}", "graph_hash": digest,
            "file_name": path.name, "file_path": str(path),
            "format": path.suffix.lower().lstrip("."),
            "source": relative.parts[0] if len(relative.parts) > 1 else "",
            "type": relative.parts[1] if len(relative.parts) > 2 else "",
            "eligible": int(not reasons), "exclusion_reason": ";".join(reasons),
            "duplicate_of": duplicate_of, **props,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-args", type=int, default=700)
    p.add_argument("--max-attacks", type=int, default=30000,
                   help="Maximum present attacks, including self-attacks (n_att)")
    args = p.parse_args()
    rows = scan_graphs(args.input_dir, args.max_args, args.max_attacks)
    columns = ["graph_id", "graph_hash", "file_name", "file_path", "format",
               "source", "type", "eligible", "exclusion_reason", "duplicate_of",
               *graph_properties(nx.DiGraph())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    duplicates = sum(bool(r["duplicate_of"]) for r in rows)
    print(f"graph files: {len(rows)} | duplicates marked: {duplicates} | "
          f"eligible: {sum(r['eligible'] for r in rows)}")
    print(args.output)


if __name__ == "__main__":
    main()
