#!/usr/bin/env python3
"""Descriptive analysis of recorded single-attack additions; no solver calls."""

import argparse
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

from parse_graphs import read_graph
from scan_graphs import graph_hash
from attack_properties import graph_cache, describe_attack, mask_arguments, make_update_id
from cad import read_args as as_set, read_cad

OUTCOMES = ["no_change", "loss_only", "gain_only", "mixed"]
CLASSES = [
    "guaranteed", "ruled_out", "unresolved_gain_ruled_out",
    "unresolved_neither_ruled_out", "unresolved_loss_ruled_out",
]
CLASS_LABELS = {
    "guaranteed": "change guaranteed",
    "ruled_out": "change ruled out",
    "unresolved_gain_ruled_out": "unresolved; gain ruled out",
    "unresolved_neither_ruled_out": "unresolved; neither direction ruled out",
    "unresolved_loss_ruled_out": "unresolved; loss ruled out",
}
DISTANCES = ["0", "1", "2", "3plus"]
AF_PROPERTIES = [
    "accepted_ratio", "n_arg", "density_ns", "self_attack_ratio",
    "reciprocal_attack_ratio_ns", "unattacked_ratio", "largest_scc_ratio",
    "cyclic_node_ratio", "condensation_depth_normalized", "reachable_pair_ratio",
]
CHECKS = [
    "gain_subset_gain_C", "gain_subset_gain_C_cad", "loss_subset_loss_C",
    "loss_subset_loss_C_cad", "delta_subset_change_C", "delta_subset_change_C_cad",
    "delta_subset_reach", "target_unaccepted_no_loss", "reciprocal_no_loss",
    "source_self_attack_no_gain", "self_addition_no_gain", "no_gain_neighbourhood",
    "change_ruled_out_no_change", "change_guaranteed_changes",
    "source_changed_cycle_closing", "unattacked_source_accepted_target_lost",
    "gain_never_contains_target",
]


def ratio(a, b):
    return a / b if b else np.nan


def theory_class(f):
    if f["change_guaranteed"] and f["change_ruled_out"]:
        raise ValueError("Update cannot be both guaranteed and ruled out")
    if f["change_guaranteed"]:
        return "guaranteed"
    if f["change_ruled_out"]:
        return "ruled_out"
    if f["gain_ruled_out"] and f["loss_ruled_out"]:
        raise ValueError("Both directions ruled out but change_ruled_out is false")
    if f["gain_ruled_out"]:
        return "unresolved_gain_ruled_out"
    if f["loss_ruled_out"]:
        return "unresolved_loss_ruled_out"
    return "unresolved_neither_ruled_out"


def theory_checks(f, r, reach, a, b, gain, loss, delta):
    c = {
        "gain_subset_gain_C": (True, not gain <= r["gain_C"]),
        "gain_subset_gain_C_cad": (True, not gain <= r["gain_C_cad"]),
        "loss_subset_loss_C": (True, not loss <= r["loss_C"]),
        "loss_subset_loss_C_cad": (True, not loss <= r["loss_C_cad"]),
        "delta_subset_change_C": (True, not delta <= r["change_C"]),
        "delta_subset_change_C_cad": (True, not delta <= r["change_C_cad"]),
        "delta_subset_reach": (True, not delta <= reach),
        "target_unaccepted_no_loss": (not f["target_acc_F"], bool(loss)),
        "reciprocal_no_loss": (f["reciprocal"], bool(loss)),
        "source_self_attack_no_gain": (f["source_self_att"], bool(gain)),
        "self_addition_no_gain": (f["self_attack_addition"], bool(gain)),
        "no_gain_neighbourhood": (f["no_gain_neighborhood_condition"], bool(gain)),
        "change_ruled_out_no_change": (f["change_ruled_out"], bool(delta)),
        "change_guaranteed_changes": (f["change_guaranteed"], not delta),
        "source_changed_cycle_closing": (a in delta, not f["cycle_closing"]),
        "unattacked_source_accepted_target_lost": (
            f["source_unatt"] and f["target_acc_F"], b not in loss),
        "gain_never_contains_target": (bool(gain), b in gain),
    }
    return {k: (int(applies), int(applies and bad)) for k, (applies, bad) in c.items()}


def read_inputs(pool_path, updates_path):
    pool = pd.read_csv(pool_path, keep_default_na=False)
    pool = pool.loc[pool.eligible.astype(str) == "1"].copy()
    updates = pd.read_csv(updates_path, dtype={"graph_id": str, "source": str, "target": str}, keep_default_na=False)
    if "update_id" not in updates:
        updates["update_id"] = [make_update_id(r.graph_id, r.source, r.target) for r in updates.itertuples(index=False)]
    if pool.graph_id.duplicated().any() or updates.update_id.duplicated().any():
        raise ValueError("Duplicate graph or update IDs")
    if updates.duplicated(["graph_id", "source", "target"]).any():
        raise ValueError("Duplicate attack pairs")
    if not set(updates.graph_id) <= set(pool.graph_id):
        raise ValueError("Unknown graph IDs in updates")
    return pool, updates


def enrich_graph(base, part, graphs_dir, cad_dir):
    path = Path(base["file_path"]) if graphs_dir is None else graphs_dir / base["file_name"]
    G = read_graph(path)
    if graph_hash(G) != base["graph_hash"]:
        raise ValueError(f"Graph differs from scanner record: {path}")
    accepted = read_cad(Path(cad_dir) / f"{base['file_name']}.dca.txt")
    if not accepted <= set(G):
        raise ValueError(f"Base CAD contains unknown arguments: {base['file_name']}")

    cache = graph_cache(G, accepted)
    distances = {x: dict(nx.single_source_shortest_path_length(G, x)) for x in G}
    records = []

    for row in part.to_dict("records"):
        a, b = row["source"], row["target"]
        if a not in G or b not in G or G.has_edge(a, b):
            raise ValueError(f"Invalid missing attack: {row['update_id']}")
        f, masks, reach_mask = describe_attack(a, b, cache)
        regions = {name: mask_arguments(mask, cache) for name, mask in masks.items()}
        reach = mask_arguments(reach_mask, cache)
        row.update(f, file_name=base["file_name"], analysis_ok=row["status"] == "completed")
        row["theory_class"] = theory_class(f)

        if row["analysis_ok"]:
            after, gain, loss, delta = (as_set(row[c]) for c in ["accepted_after_args", "gain_args", "loss_args", "delta_args"])
            counts = [len(accepted), len(after), len(gain), len(loss), len(delta)]
            given = [int(row[c]) for c in ["accepted_count_before", "accepted_count_after", "gain_count", "loss_count", "delta_count"]]
            if gain & loss or gain != after - accepted or loss != accepted - after or delta != gain | loss or counts != given:
                raise ValueError(f"Inconsistent acceptance sets/counts: {row['update_id']}")
            row.update(
                any_change=bool(delta), any_gain=bool(gain), any_loss=bool(loss),
                outcome="mixed" if gain and loss else "gain_only" if gain else "loss_only" if loss else "no_change",
            )
            for name, (applies, bad) in theory_checks(row, regions, reach, a, b, gain, loss, delta).items():
                row[f"check_{name}_applicable"] = applies
                row[f"check_{name}_violated"] = bad
            for d in DISTANCES:
                row[f"distance_{d}_candidates"] = row[f"distance_{d}_changed"] = 0
            if a != b:
                for x in regions["change_C"]:
                    d = distances[b][x]
                    k = str(d) if d <= 2 else "3plus"
                    row[f"distance_{k}_candidates"] += 1
                    row[f"distance_{k}_changed"] += int(x in delta)
        records.append(row)
    return pd.DataFrame(records), accepted


def graph_summary(base, rows, accepted):
    data = rows.loc[rows.analysis_ok].copy()
    nonself = data.loc[data.source != data.target]
    out = dict(base)
    out.update(
        n_expected_updates=int(out["non_att"]), n_recorded_updates=len(rows),
        n_completed_updates=len(data), enumeration_complete=len(data) == int(out["non_att"]),
        accepted_count=len(accepted), accepted_ratio=len(accepted) / int(out["n_arg"]),
    )
    for scope, part in [("all", data), ("nonself", nonself)]:
        out[f"{scope}_n_updates"] = len(part)
        out[f"{scope}_change_rate"] = ratio(int(part.any_change.sum()), len(part))

    total = len(data) * int(out["n_arg"])
    out["share_reach"] = ratio(data.target_reach_count.sum(), total)
    out["share_change_C"] = ratio(data.change_C_size.sum(), total)
    out["share_change_C_cad"] = ratio(data.change_C_cad_size.sum(), total)
    for d in DISTANCES:
        cand = int(nonself[f"distance_{d}_candidates"].sum())
        changed = int(nonself[f"distance_{d}_changed"].sum())
        out[f"distance_{d}_candidates"] = cand
        out[f"distance_{d}_changed"] = changed
        out[f"distance_{d}_rate"] = ratio(changed, cand)
    return out


def category_row(part, scope, category, label):
    changed = part.loc[part.any_change.astype(bool)]
    rates = part.groupby("graph_id").any_change.mean()
    row = dict(
        scope=scope, category=category, label=label, n_graphs=int(part.graph_id.nunique()),
        n_updates=len(part), n_self=int((part.source == part.target).sum()),
        n_nonself=int((part.source != part.target).sum()),
        pooled_change_percent=100 * ratio(int(part.any_change.sum()), len(part)),
        median_AF_change_percent=100 * rates.median() if len(rates) else np.nan,
        mean_delta_when_changed=changed.delta_count.mean() if len(changed) else np.nan,
    )
    row.update({o: int((part.outcome == o).sum()) for o in OUTCOMES})
    return row


def make_tables(updates, graphs):
    data = updates.loc[updates.analysis_ok].copy()
    changed = data.loc[data.any_change.astype(bool)]

    # 01 study / outcomes
    rows = []
    def add(scope, measure, value, unit="count"):
        rows.append(dict(scope=scope, measure=measure, value=value, unit=unit))
    add("study", "frameworks", len(graphs))
    for c in ["n_arg", "n_att"]:
        for name, value in [("minimum", graphs[c].min()), ("median", graphs[c].median()), ("maximum", graphs[c].max())]:
            add("study", f"{c}_{name}", value)
    add("study", "completed_updates", len(data)); add("study", "nonself_updates", int((data.source != data.target).sum())); add("study", "self_updates", int((data.source == data.target).sum()))
    for o in OUTCOMES:
        n = int((data.outcome == o).sum()); add("outcomes", o, n); add("outcomes", f"{o}_share", 100*n/len(data), "percent")
    add("outcomes", "changed_updates", len(changed)); add("outcomes", "changed_updates_share", 100*len(changed)/len(data), "percent")
    add("outcomes", "loss_share_among_changed_updates", 100*ratio(int(data.any_loss.sum()), len(changed)), "percent")
    add("extent", "mean_delta_when_changed", changed.delta_count.mean(), "arguments"); add("extent", "median_delta_when_changed", changed.delta_count.median(), "arguments"); add("extent", "maximum_delta", changed.delta_count.max(), "arguments")
    add("extent", "singleton_share_when_changed", 100*ratio(int((changed.delta_count == 1).sum()), len(changed)), "percent"); add("extent", "at_most_two_share_when_changed", 100*ratio(int((changed.delta_count <= 2).sum()), len(changed)), "percent")
    pairs = int(data.n_arg.sum()); add("argument_update_pairs", "all", pairs); add("argument_update_pairs", "changed", int(data.delta_count.sum())); add("argument_update_pairs", "changed_share", 100*ratio(int(data.delta_count.sum()), pairs), "percent"); add("argument_update_pairs", "gain_total", int(data.gain_count.sum())); add("argument_update_pairs", "loss_total", int(data.loss_count.sum()))
    tables = {"01_study_and_outcomes": pd.DataFrame(rows)}

    # 02 theory classes: all + non-self
    rows = []
    for scope, part in [("all_additions", data), ("nonself_additions", data.loc[data.source != data.target])]:
        for c in CLASSES:
            rows.append(category_row(part.loc[part.theory_class == c], scope, c, CLASS_LABELS[c]))
    tables["02_theory_classes"] = pd.DataFrame(rows)

    # 03 endpoint status: non-self only
    labels = {"00": "source unaccepted; target unaccepted", "01": "source unaccepted; target accepted", "10": "source accepted; target unaccepted", "11": "source accepted; target accepted"}
    nonself = data.loc[data.source != data.target]
    tables["03_endpoint_status"] = pd.DataFrame([category_row(nonself.loc[nonself.endpoint_status == s], "nonself", s, labels[s]) for s in labels])

    # 04 self-additions
    self_data = data.loc[data.source == data.target]
    tables["04_self_additions"] = pd.DataFrame([
        category_row(self_data.loc[self_data.source_acc_F.astype(bool) == val], "self", "accepted" if val else "unaccepted", "initially accepted" if val else "initially unaccepted")
        for val in [False, True]
    ])

    # 05 nested candidate bounds, all relative to A
    total = int(data.n_arg.sum())
    specs = [("A", None, "All arguments", 1.0), ("Reach", "target_reach_count", "Reach_F(b)", graphs.share_reach.median()), ("C_delta", "change_C_size", "C_e^Delta", graphs.share_change_C.median()), ("C_cad_delta", "change_C_cad_size", "C_cad,e^Delta", graphs.share_change_C_cad.median())]
    rows = []
    for name, col, label, median_share in specs:
        n = total if col is None else int(data[col].sum())
        share = 100 * ratio(n, total)
        rows.append(dict(region=name, label=label, argument_update_occurrences=n, pooled_retained_percent_of_A=share, pooled_reduction_from_A_percent=100-share, median_AF_retained_percent_of_A=100*median_share))
    tables["05_candidate_regions"] = pd.DataFrame(rows)

    # 06 distance inside C_delta, non-self only
    rows = []
    for d in DISTANCES:
        cand = int(graphs[f"distance_{d}_candidates"].sum()); ch = int(graphs[f"distance_{d}_changed"].sum()); rates = graphs[f"distance_{d}_rate"].dropna()
        rows.append(dict(distance_from_target="3+" if d == "3plus" else d, n_graphs_with_candidates=len(rates), candidate_occurrences=cand, changed_occurrences=ch, pooled_change_percent=100*ratio(ch, cand), median_AF_change_percent=100*rates.median() if len(rates) else np.nan))
    tables["06_distance_to_target"] = pd.DataFrame(rows)

    # 07 pre-specified AF-level Spearman correlations with overall change rate
    rows = []
    for prop in AF_PROPERTIES:
        if prop in graphs.columns:
            pair = graphs[[prop, "all_change_rate"]].dropna()
            rows.append(dict(property=prop, n_graphs=len(pair), spearman_rho=pair[prop].rank().corr(pair.all_change_rate.rank()) if len(pair) > 1 else np.nan))
    tables["07_af_property_spearman"] = pd.DataFrame(rows)

    # 08 figure; non-self rate is included only as a sensitivity check
    tables["08_acceptance_vs_change"] = graphs[["graph_id", "file_name", "accepted_ratio", "all_n_updates", "all_change_rate", "nonself_n_updates", "nonself_change_rate"]].copy()
    return tables


def theory_audit(data):
    return pd.DataFrame([
        dict(check=name, n_applicable=int(data[f"check_{name}_applicable"].sum()), n_violations=int(data[f"check_{name}_violated"].sum()))
        for name in CHECKS
    ])


def save_figure(graphs, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.scatter(100*graphs.accepted_ratio, 100*graphs.all_change_rate, s=30, alpha=.9)
    ax.set_xlabel("Initial credulous-acceptance ratio per AF (%)")
    ax.set_ylabel("Acceptance-change rate per AF (%)")
    ax.xaxis.set_major_formatter(PercentFormatter(100)); ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102); fig.tight_layout()
    fig.savefig(directory / "01_acceptance_and_change.png", dpi=190, bbox_inches="tight")
    fig.savefig(directory / "01_acceptance_and_change.svg", bbox_inches="tight"); plt.close(fig)


def run(pool_path, updates_path, output, cad_dir, graphs_dir=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    pool, updates = read_inputs(pool_path, updates_path)
    update_frames, graph_rows = [], []
    for base in pool.to_dict("records"):
        part = updates.loc[updates.graph_id == base["graph_id"]]
        if part.empty:
            raise ValueError(f"No recorded updates for {base['file_name']}")
        enriched, accepted = enrich_graph(base, part, graphs_dir, cad_dir)
        update_frames.append(enriched); graph_rows.append(graph_summary(base, enriched, accepted))

    all_updates = pd.concat(update_frames, ignore_index=True); all_graphs = pd.DataFrame(graph_rows)
    all_updates.to_csv(output / "exhaustive_updates.csv", index=False); all_graphs.to_csv(output / "exhaustive_graphs.csv", index=False)
    data = all_updates.loc[all_updates.analysis_ok].copy()
    if data.empty:
        raise ValueError("No completed updates")

    table_dir = output / "tables"; table_dir.mkdir(exist_ok=True)
    for name, table in make_tables(all_updates, all_graphs).items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    audit_dir = output / "audit"; audit_dir.mkdir(exist_ok=True)
    audit = theory_audit(data); audit.to_csv(audit_dir / "theory_checks.csv", index=False)
    save_figure(all_graphs, output / "figures")
    print(f"AFs: {len(all_graphs)} | completed updates: {len(data):,} | complete enumerations: {int(all_graphs.enumeration_complete.sum())}/{len(all_graphs)}")
    print(f"Theory violations: {int(audit.n_violations.sum())}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, required=True); p.add_argument("--updates", type=Path, required=True)
    p.add_argument("--base-cad-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--graphs-dir", type=Path); a = p.parse_args()
    run(a.pool, a.updates, a.output_dir, a.base_cad_dir, a.graphs_dir)


if __name__ == "__main__":
    main()
