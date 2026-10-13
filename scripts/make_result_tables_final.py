from pathlib import Path
import pandas as pd

INPUT = Path("all_results.csv")
OUTPUT_DIR = Path("result_tables")


def write_table(df, path):
    """Write full-precision analysis CSV and a 3-decimal publication copy."""
    df.to_csv(path, index=False)

    table = df.copy()
    float_cols = table.select_dtypes(include="float").columns
    table[float_cols] = table[float_cols].round(3)

    table_path = path.with_name(path.stem + "_table.csv")
    table.to_csv(table_path, index=False)


def delta_table(left, right, keys, label_left, label_right):
    """Join controlled conditions; report AP/MCC deltas and diagnostic rates."""
    metrics = [
        "test_ap",
        "test_mcc",
        "test_precision",
        "test_recall",
        "test_specificity",
    ]
    a = left[keys + metrics].copy()
    b = right[keys + metrics].copy()

    a = a.rename(columns={m: f"{label_left}_{m[5:]}" for m in metrics})
    b = b.rename(columns={m: f"{label_right}_{m[5:]}" for m in metrics})

    out = a.merge(b, on=keys, validate="one_to_one")
    out["delta_ap"] = out[f"{label_right}_ap"] - out[f"{label_left}_ap"]
    out["delta_mcc"] = out[f"{label_right}_mcc"] - out[f"{label_left}_mcc"]
    return out


def main():
    df = pd.read_csv(INPUT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Specificity/TNR is not stored in every summary, but is determined by TN and FP.
    if "test_specificity" not in df:
        df["test_specificity"] = (
            df["test_tn"] / (df["test_tn"] + df["test_fp"])
        )

    # Normalize names used only for presentation.
    df["feature_set"] = df["feature_set"].replace({
        "rf_struct": "struct",
        "rf_struct_cad": "struct_cad",
        "struct+cad": "struct_cad",
    })

    # ------------------------------------------------------------------
    # RQ1: overall prediction performance
    # ------------------------------------------------------------------
    rq1_cols = [
        "run", "model", "feature_set", "train_data",
        "aux_direction", "c_loss_mask", "n_estimators",
        "test_ap", "test_mcc", "test_precision", "test_recall", "test_specificity",
        "test_cfilter_fp_reduced", "test_cfilter_mcc",
    ]
    rq1 = df[[c for c in rq1_cols if c in df.columns]].copy()
    write_table(rq1, OUTPUT_DIR / "rq1_overview.csv")

    # For controlled GINE feature ablations, keep direction and C-loss mask off.
    base = df[
        (df["model"] == "GINE")
        & (~df["aux_direction"].astype(bool))
        & (~df["c_loss_mask"].astype(bool))
    ].copy()

    # ------------------------------------------------------------------
    # RQ2: effect of initial CAD status
    # basic -> basic_cad, struct -> struct_cad
    # ------------------------------------------------------------------
    rq2_parts = []
    for base_feature, cad_feature in [("basic", "basic_cad"), ("struct", "struct_cad")]:
        left = base[base["feature_set"] == base_feature].copy()
        right = base[base["feature_set"] == cad_feature].copy()
        t = delta_table(
            left, right, ["train_data"],
            "without_cad", "with_cad"
        )
        t.insert(1, "base_features", base_feature)
        rq2_parts.append(t)

    rq2 = pd.concat(rq2_parts, ignore_index=True)
    write_table(rq2, OUTPUT_DIR / "rq2_cad.csv")

    # ------------------------------------------------------------------
    # RQ3: effect of theory-guided structural features
    # basic -> struct; basic_cad -> struct_cad
    # ------------------------------------------------------------------
    rq3_parts = []
    for cad_input, basic_feature, struct_feature in [
        (False, "basic", "struct"),
        (True, "basic_cad", "struct_cad"),
    ]:
        left = base[base["feature_set"] == basic_feature].copy()
        right = base[base["feature_set"] == struct_feature].copy()
        t = delta_table(
            left, right, ["train_data"],
            "basic", "struct"
        )
        t.insert(1, "cad_input", cad_input)
        rq3_parts.append(t)

    rq3 = pd.concat(rq3_parts, ignore_index=True)
    write_table(rq3, OUTPUT_DIR / "rq3_structure.csv")

    # ------------------------------------------------------------------
    # RQ5: auxiliary gain/loss direction supervision
    # compare within same feature set and training data, without C-loss mask
    # ------------------------------------------------------------------
    d = df[(df["model"] == "GINE") & (~df["c_loss_mask"].astype(bool))].copy()
    no_dir = d[~d["aux_direction"].astype(bool)]
    yes_dir = d[d["aux_direction"].astype(bool)]
    common_features = sorted(set(no_dir["feature_set"]) & set(yes_dir["feature_set"]))

    rq5_parts = []
    for feature in common_features:
        left = no_dir[no_dir["feature_set"] == feature]
        right = yes_dir[yes_dir["feature_set"] == feature]
        common_train = sorted(set(left["train_data"]) & set(right["train_data"]))
        if not common_train:
            continue
        left = left[left["train_data"].isin(common_train)]
        right = right[right["train_data"].isin(common_train)]
        t = delta_table(
            left, right, ["train_data"],
            "no_direction", "direction"
        )
        t.insert(1, "feature_set", feature)
        rq5_parts.append(t)

    rq5 = pd.concat(rq5_parts, ignore_index=True)
    write_table(rq5, OUTPUT_DIR / "rq5_direction.csv")

    # ------------------------------------------------------------------
    # Additional analysis: exhaustive-training augmentation
    # no direction, no C-loss mask
    # ------------------------------------------------------------------
    bench = base[base["train_data"] == "benchmark"]
    exh = base[base["train_data"] == "benchmark+exhaustive"]
    exhaustive = delta_table(
        bench, exh, ["feature_set"],
        "benchmark", "with_exhaustive"
    )
    write_table(exhaustive, OUTPUT_DIR / "exhaustive_ablation.csv")

    # ------------------------------------------------------------------
    # Additional analysis: C-loss-mask
    # Only configurations where both mask states actually exist.
    # ------------------------------------------------------------------
    g = df[df["model"] == "GINE"].copy()
    no_mask = g[~g["c_loss_mask"].astype(bool)]
    with_mask = g[g["c_loss_mask"].astype(bool)]

    keys = ["feature_set", "train_data", "aux_direction"]
    common = no_mask[keys].merge(with_mask[keys], on=keys).drop_duplicates()

    if len(common):
        left = no_mask.merge(common, on=keys)
        right = with_mask.merge(common, on=keys)
        metrics = ["test_ap", "test_mcc", "test_precision", "test_recall", "test_specificity"]
        left = left[keys + metrics].rename(
            columns={m: f"no_loss_mask_{m[5:]}" for m in metrics}
        )
        right = right[keys + metrics].rename(
            columns={m: f"loss_mask_{m[5:]}" for m in metrics}
        )
        loss_mask = left.merge(right, on=keys, validate="one_to_one")
        loss_mask["delta_ap"] = loss_mask["loss_mask_ap"] - loss_mask["no_loss_mask_ap"]
        loss_mask["delta_mcc"] = loss_mask["loss_mask_mcc"] - loss_mask["no_loss_mask_mcc"]
    else:
        loss_mask = pd.DataFrame()

    write_table(loss_mask, OUTPUT_DIR / "c_loss_mask_ablation.csv")

    # ------------------------------------------------------------------
    # Additional analysis: deterministic C_delta prediction filter
    # ------------------------------------------------------------------
    cfilter_cols = [
        "run", "feature_set", "train_data", "aux_direction", "c_loss_mask",
        "test_fp", "test_cfilter_fp", "test_cfilter_fp_reduced",
        "test_mcc", "test_cfilter_mcc",
    ]
    cfilter = df[
        (df["model"] == "GINE")
        & df["test_cfilter_fp"].notna()
    ][cfilter_cols].copy()
    cfilter["delta_mcc"] = cfilter["test_cfilter_mcc"] - cfilter["test_mcc"]
    write_table(cfilter, OUTPUT_DIR / "cfilter_effect.csv")

    print("Written:")
    for path in sorted(OUTPUT_DIR.glob("*_table.csv")):
        print(" ", path)


if __name__ == "__main__":
    main()
