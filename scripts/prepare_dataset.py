#!/usr/bin/env python3
"""Split AFs, sample 7+1+1+1 updates, write all updated graphs, then compute CAD.

Use current candidate files and a new output directory for the pooled policy.
Reruns keep the saved split and attacks, reusing complete CAD files.
"""
import argparse
import hashlib
import random
import shutil
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from attack_properties import make_update_id
from cad import compute_cad, read_cad, join_args
from parse_graphs import read_graph

SPLITS = ["train", "val", "test"]
HIGH = ["unresolved_gain_ruled_out", "unresolved_neither_ruled_out"]
LOW = "unresolved_loss_ruled_out"
CONTROLS = ["guaranteed", "ruled_out"]
N_PER_GRAPH, SELF_RATE = 10, 0.20
RESULT_COLUMNS = ["runtime_seconds", "cad_origin", "timeout_argument", "accepted_count_before",
                  "accepted_count_after", "accepted_after_args", "gain_count", "loss_count",
                  "delta_count", "gain_args", "loss_args", "delta_args", "outcome_type"]
INSTANCE_COLUMNS = ["graph_id", "file_name", "source", "target", "delta_args", "gain_args", "loss_args"]


def rng_for(seed, *parts):
    key = "".join(f"{len(str(x))}:{x}" for x in (seed, *parts))
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))


def read_table(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_table(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def split_graphs(pool, seed):
    pool = pool.sort_values("graph_id").reset_index(drop=True).copy()
    pool["size_group"] = pd.qcut(pool.n_arg.astype(int), 3, labels=["small", "medium", "large"])
    pool["acceptance_group"] = pd.cut(pool.accepted_ratio.astype(float),
                                      [-1e-9, 0.2, 0.6, 1.0000001], right=False, labels=["low", "medium", "high"])
    strata = pool.size_group.astype(str) + "_" + pool.acceptance_group.astype(str)
    first = StratifiedShuffleSplit(n_splits=1, test_size=round(.25 * len(pool)), random_state=seed)
    rest, test = next(first.split(pool, strata))
    second = StratifiedShuffleSplit(n_splits=1, test_size=round(.10 * len(pool)), random_state=seed)
    _, val = next(second.split(pool.iloc[rest], strata.iloc[rest]))
    pool["split"] = "train"
    pool.loc[rest[val], "split"] = "val"
    pool.loc[test, "split"] = "test"
    return pool


def prepare_splits(args):
    path = args.output_dir / "split_graphs.csv"
    if path.exists():
        return read_table(path)
    pool = read_table(args.pool)
    summary = read_table(args.candidates_dir / "candidate_summary.csv")
    details = [c for c in summary if c == "graph_id" or c not in pool]
    pool = pool.merge(summary[details], on="graph_id", how="left", validate="one_to_one")
    pool["accepted_count"] = [len(read_cad(args.base_cad_dir / f"{name}.dca.txt"))
                              for name in pool.file_name]
    pool["accepted_ratio"] = pool.accepted_count / pool.n_arg.astype(int)
    pool["file_path"] = [str((args.root / p).resolve()) for p in pool.file_path]
    pool = split_graphs(pool, args.seed)
    save_table(pool, path)
    for split in SPLITS:
        save_table(pool.loc[pool.split == split], args.output_dir / f"{split}_graphs.csv")
    settings = {"seed": args.seed, "updates_per_graph": N_PER_GRAPH, "self_rate": SELF_RATE,
                "test_fraction": .25, "val_fraction": .10, "policy": "pooled_loss_10"}
    save_table(pd.DataFrame(settings.items(), columns=["setting", "value"]),
               args.output_dir / "settings.csv")
    return pool


def allocate_self_slots(graphs, seed, split):
    # First reserve the loops needed when there are too few non-self candidates.
    slots = {r.graph_id: max(0, min(N_PER_GRAPH, int(r.n_candidates))
                             - (int(r.n_candidates) - int(r.n_self)))
             for r in graphs.itertuples(index=False)}
    available = sorted(r.graph_id for r in graphs.itertuples(index=False)
                       if int(r.n_self) > 0 and slots[r.graph_id] == 0)
    extra = min(max(0, round(SELF_RATE * len(graphs)) - sum(slots.values())), len(available))
    for gid in rng_for(seed, split, "self_allocation").sample(available, extra):
        slots[gid] = 1
    return slots


def sample_attacks(candidates, graph_id, n_self, seed):
    candidates = candidates.sort_values(["source", "target"]).reset_index(drop=True)
    if len(candidates) <= N_PER_GRAPH:
        return [("all_missing", row) for row in candidates.to_dict("records")]
    nonself = candidates.loc[candidates.source != candidates.target]
    selected, used = [], set()
    rng = rng_for(seed, graph_id, "attack_draws")

    def draw(pool, count, role):
        available = [i for i in pool.index if i not in used]
        count = min(count, len(available), N_PER_GRAPH - len(selected))
        for i in rng.sample(available, count):
            used.add(i)
            selected.append((role, candidates.loc[i].to_dict()))

    draw(candidates.loc[candidates.source == candidates.target], n_self, "self_control")
    covered = {row["sampling_class"] for _, row in selected}
    for category in CONTROLS:
        if category not in covered:
            draw(nonself.loc[nonself.sampling_class == category], 1, category)
    # One uniform draw from the UNION, with no quota for its two subtypes.
    draw(nonself.loc[nonself.sampling_class.isin(HIGH)], 7, "unresolved_loss_not_ruled_out")
    draw(nonself.loc[nonself.sampling_class == LOW], 1, LOW)
    for categories, role in [(HIGH, "fallback_loss_not_ruled_out"),
                             ([LOW], "fallback_loss_ruled_out"),
                             (CONTROLS, "fallback_control")]:
        draw(nonself.loc[nonself.sampling_class.isin(categories)],
             N_PER_GRAPH - len(selected), role)
    return selected


def updated_name(file_name, source, target):
    path = Path(file_name)
    return f"{path.stem}__add_{source}_{target}{path.suffix}"


def prepare_updates(graphs, split, args):
    """Sample once and create all selected F+e files; reruns restore missing files."""
    directory = args.output_dir / split
    path = directory / "updates.csv"
    for folder in ["base_graphs", "base_cad", "updated_graphs", "updated_cad"]:
        (directory / folder).mkdir(parents=True, exist_ok=True)
    saved = read_table(path) if path.exists() else None
    self_slots = allocate_self_slots(graphs, args.seed, split) if saved is None else {}
    records = []
    for base in graphs.itertuples(index=False):
        graph_path = directory / "base_graphs" / base.file_name
        cad_path = directory / "base_cad" / f"{base.file_name}.dca.txt"
        for source, destination in [(Path(base.file_path), graph_path),
                                    (args.base_cad_dir / cad_path.name, cad_path)]:
            if not destination.exists():
                shutil.copy2(source, destination)
        if saved is None:
            candidates = read_table(args.candidates_dir / "by_graph" / base.candidate_file)
            chosen = sample_attacks(candidates, base.graph_id, self_slots[base.graph_id], args.seed)
            rows = []
            for rank, (role, attack) in enumerate(chosen, 1):
                a, b = attack["source"], attack["target"]
                rows.append({**attack, "update_id": make_update_id(base.graph_id, a, b),
                             "graph_id": base.graph_id, "file_name": base.file_name, "split": split,
                             "sample_rank": rank, "sample_role": role, "sampling_seed": args.seed,
                             "sampling_scheme": "theory_pilot_stratified_pooled",
                             "updated_file": updated_name(base.file_name, a, b), "status": "pending",
                             **{column: "" for column in RESULT_COLUMNS}})
        else:
            rows = saved.loc[saved.graph_id == base.graph_id].to_dict("records")
        graph = read_graph(graph_path)
        for row in rows:
            destination = directory / "updated_graphs" / row["updated_file"]
            if not destination.exists():
                write_updated_graph(graph, row["source"], row["target"], destination)
        records.extend(rows)
    updates = pd.DataFrame(records, columns=list(records[0]) if records else
    INSTANCE_COLUMNS + ["status", "updated_file"])
    save_table(updates, path)  # All splits are prepared before the first solver call.
    save_table(updates.loc[updates.status == "completed", INSTANCE_COLUMNS], directory / "instances.csv")
    print(f"{split}: {len(graphs)} AFs, {len(updates)} selected additions, "
          f"{sum(updates.source == updates.target)} self-additions")


def write_updated_graph(graph, source, target, path):
    updated = graph.copy()
    updated.add_edge(source, target)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if path.suffix.lower() == ".tgf":
            handle.writelines(f"{x}\n" for x in updated)
            handle.write("#\n")
            handle.writelines(f"{a} {b}\n" for a, b in updated.edges())
        else:  # The parser/input preparation allows only TGF and APX.
            handle.writelines(f"arg({x}).\n" for x in updated)
            handle.writelines(f"att({a},{b}).\n" for a, b in updated.edges())
    temporary.replace(path)


def label_update(before, after):
    gain = after - before
    loss = before - after
    delta = gain | loss
    if not delta:
        outcome = "no_change"
    elif gain and loss:
        outcome = "mixed"
    elif gain:
        outcome = "gain_only"
    else:
        outcome = "loss_only"
    return {"accepted_count_before": len(before), "accepted_count_after": len(after),
            "accepted_after_args": join_args(after), "gain_count": len(gain),
            "loss_count": len(loss), "delta_count": len(delta),
            "gain_args": join_args(gain), "loss_args": join_args(loss),
            "delta_args": join_args(delta), "outcome_type": outcome}


def compute_updates(split, args):
    """Read each base CAD once; compute/reuse only unfinished or missing updated CAD."""
    directory = args.output_dir / split
    path = directory / "updates.csv"
    updates = read_table(path)
    for _, part in updates.groupby("graph_id", sort=False):
        missing_cad = part.updated_file.map(
            lambda name: not (directory / "updated_cad" / f"{name}.dca.txt").exists())
        pending = part.loc[(part.status != "completed") | missing_cad]
        if pending.empty:
            continue
        before = read_cad(directory / "base_cad" / f"{part.file_name.iloc[0]}.dca.txt")
        for index, row in pending.iterrows():
            graph_path = directory / "updated_graphs" / row.updated_file
            cad_path = directory / "updated_cad" / f"{row.updated_file}.dca.txt"
            start = perf_counter()
            after, failed, computed = compute_cad(graph_path, args.solver, args.timeout, cad_path)
            elapsed = round(perf_counter() - start, 6)
            result = {column: "" for column in RESULT_COLUMNS}
            result.update(accepted_count_before=len(before), timeout_argument=failed or "",
                          runtime_seconds=elapsed if computed or after is None else "",
                          status="timeout" if after is None else "completed")
            if after is not None:
                result.update(label_update(before, after), cad_origin="computed" if computed else "available")
            for column, value in result.items():
                updates.at[index, column] = str(value)
        save_table(updates, path)  # Save completed
        save_table(updates.loc[updates.status == "completed", INSTANCE_COLUMNS], directory / "instances.csv")
        print(f"{split}: {sum(updates.status == 'completed')}/{len(updates)} completed")
    save_table(updates.loc[updates.status == "completed", INSTANCE_COLUMNS], directory / "instances.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--base-cad-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    graphs = prepare_splits(args)
    # Phase 1: sample and create every updated graph before computing labels.
    for split in SPLITS:
        prepare_updates(graphs.loc[graphs.split == split], split, args)
    # Phase 2: complete the selected updates graph by graph.
    for split in SPLITS:
        compute_updates(split, args)


if __name__ == "__main__":
    main()
