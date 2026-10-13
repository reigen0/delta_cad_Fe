#!/usr/bin/env python3
"""Evaluate per-AF prediction quality against graph properties."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, recall_score
from statsmodels.nonparametric.smoothers_lowess import lowess


PROPERTIES = [
    "n_arg",
    "density_ns",
    "self_attack_ratio",
    "reciprocal_attack_ratio_ns",
    "in_degree_cv_ns",
    "out_degree_cv_ns",
    "largest_scc_ratio",
    "reachable_pair_ratio",
]

PLOT_PROPERTIES = [
    "density_ns",
    "largest_scc_ratio",
    "reachable_pair_ratio",
    "in_degree_cv_ns",
]

PROPERTY_LABELS = {
    "density_ns": "Attack density excluding self-attacks",
    "largest_scc_ratio": "Largest SCC ratio",
    "reachable_pair_ratio": "Reachable-pair ratio",
    "in_degree_cv_ns": "In-degree CV excluding self-attacks",
}

#calculate metrics for evaluation
def graph_metrics(path, suffix):
    pred = pd.read_csv(path)
    rows = []

    for graph_id, g in pred.groupby("graph_id"):
        y = g["y_true"].to_numpy()
        p = g["y_pred"].to_numpy()

        has_positive = bool((y == 1).any())
        has_negative = bool((y == 0).any())

        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fn = int(((y == 1) & (p == 0)).sum())

        rows.append({
            "graph_id": graph_id,
            f"mcc_{suffix}": (
                matthews_corrcoef(y, p)
                if has_positive and has_negative
                else np.nan
            ),
            f"recall_{suffix}": (
                recall_score(y, p)
                if has_positive
                else np.nan
            ),
            f"specificity_{suffix}": (
                tn / (tn + fp)
                if has_negative
                else np.nan
            ),
            f"tp_{suffix}": tp,
            f"fp_{suffix}": fp,
            f"tn_{suffix}": tn,
            f"fn_{suffix}": fn,
            f"perfect_no_change_{suffix}": (
                int(p.sum() == 0)
                if not has_positive
                else np.nan
            ),
            "n_argument_instances": len(g),
            "n_positive": int(y.sum()),
            "positive_rate": y.mean(),
            "is_no_change_graph": int(not has_positive),
        })

    return pd.DataFrame(rows)


def rho(x, y):
    return spearmanr(x, y, nan_policy="omit").statistic

#generate plot
def save_scatterplots(graph_level, output_dir):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for prop in PLOT_PROPERTIES:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(10.4, 4.2),
            sharex=True,
            sharey=True,
        )

        for ax, metric, title in [
            (axes[0], "mcc_struct", "struct"),
            (axes[1], "mcc_struct_cad", "struct_cad"),
        ]:
            valid = graph_level[[prop, metric]].dropna().sort_values(prop)
            x = valid[prop].to_numpy()
            y = valid[metric].to_numpy()

            r = rho(x, y)

            ax.scatter(
                x,
                y,
                s=20,
                alpha=0.60,
            )

            # LOWESS is only a visual guide to the monotone trend.
            if len(valid) >= 8 and np.unique(x).size >= 4:
                smooth = lowess(y, x, frac=0.55, return_sorted=True)
                ax.plot(
                    smooth[:, 0],
                    smooth[:, 1],
                    linewidth=1.6,
                )

            ax.text(
                0.04,
                0.95,
                f"Spearman ρ = {r:.3f}\nn = {len(valid)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
            )

            ax.set_title(title)
            ax.set_xlabel(PROPERTY_LABELS[prop])
            ax.grid(alpha=0.18)

        axes[0].set_ylabel("Per-AF MCC")

        fig.suptitle(
            f"{PROPERTY_LABELS[prop]} vs. prediction performance",
            y=1.02,
        )
        fig.tight_layout()

        fig.savefig(
            figure_dir / f"{prop}_vs_mcc.png",
            dpi=240,
            bbox_inches="tight",
        )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--struct", type=Path, required=True)
    p.add_argument("--struct_cad", type=Path, required=True)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    struct = graph_metrics(args.struct, "struct")

    cad = graph_metrics(args.struct_cad, "struct_cad").drop(
        columns=[
            "n_argument_instances",
            "n_positive",
            "positive_rate",
            "is_no_change_graph",
        ]
    )

    graph_info = pd.read_csv(args.graphs)[
        ["graph_id", "file_name"] + PROPERTIES
    ]

    graph_level = struct.merge(
        cad,
        on="graph_id",
        validate="one_to_one",
    ).merge(
        graph_info,
        on="graph_id",
        validate="one_to_one",
    )

    rows = []
    for prop in PROPERTIES:
        rows.append({
            "property": prop,
            "rho_mcc_struct": rho(
                graph_level[prop],
                graph_level["mcc_struct"],
            ),
            "rho_mcc_struct_cad": rho(
                graph_level[prop],
                graph_level["mcc_struct_cad"],
            ),
            "rho_recall_struct": rho(
                graph_level[prop],
                graph_level["recall_struct"],
            ),
            "rho_recall_struct_cad": rho(
                graph_level[prop],
                graph_level["recall_struct_cad"],
            ),
            "rho_specificity_struct": rho(
                graph_level[prop],
                graph_level["specificity_struct"],
            ),
            "rho_specificity_struct_cad": rho(
                graph_level[prop],
                graph_level["specificity_struct_cad"],
            ),
            "rho_positive_rate": rho(
                graph_level[prop],
                graph_level["positive_rate"],
            ),
        })

    correlations = pd.DataFrame(rows)

    no_change = graph_level.loc[
        graph_level["is_no_change_graph"] == 1
    ]
    no_change_summary = pd.DataFrame([{
        "n_no_change_graphs": len(no_change),
        "perfect_no_change_struct":
            int(no_change["perfect_no_change_struct"].sum()),
        "perfect_no_change_struct_cad":
            int(no_change["perfect_no_change_struct_cad"].sum()),
    }])

    diagnostic_columns = [
        "file_name",
        "graph_id",
        "n_argument_instances",
        "n_positive",
        "positive_rate",
        "mcc_struct",
        "recall_struct",
        "specificity_struct",
        "tp_struct",
        "fp_struct",
        "tn_struct",
        "fn_struct",
        "mcc_struct_cad",
        "recall_struct_cad",
        "specificity_struct_cad",
        "tp_struct_cad",
        "fp_struct_cad",
        "tn_struct_cad",
        "fn_struct_cad",
    ]
    diagnostics = graph_level[diagnostic_columns].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Full-precision analysis files
    graph_level.to_csv(
        args.output_dir / "graph_level_metrics_exh.csv",
        index=False,
    )
    correlations.to_csv(
        args.output_dir / "graph_property_correlations_exh.csv",
        index=False,
    )
    diagnostics.to_csv(
        args.output_dir / "graph_prediction_diagnostics_exh.csv",
        index=False,
    )
    no_change_summary.to_csv(
        args.output_dir / "no_change_graph_summary_exh.csv",
        index=False,
    )

    # Rounded table versions
    correlations.round(3).to_csv(
        args.output_dir / "graph_property_correlations_table_exh.csv",
        index=False,
    )

    diagnostics_table = diagnostics.copy()
    float_cols = diagnostics_table.select_dtypes(
        include="float"
    ).columns
    diagnostics_table[float_cols] = (
        diagnostics_table[float_cols].round(3)
    )
    diagnostics_table.to_csv(
        args.output_dir / "graph_prediction_diagnostics_table_exh.csv",
        index=False,
    )

    save_scatterplots(graph_level, args.output_dir)

    print(args.output_dir / "graph_level_metrics.csv")
    print(args.output_dir / "graph_property_correlations.csv")
    print(args.output_dir / "graph_prediction_diagnostics.csv")
    print(args.output_dir / "figures")


if __name__ == "__main__":
    main()
