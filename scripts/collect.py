from pathlib import Path
import pandas as pd

GINE_DIR = Path("results_gine")
RF_DIR = Path("results_rf")
OUTPUT = Path("all_results.csv")

rows = []

for model, folder in [("GINE", GINE_DIR), ("RF", RF_DIR)]:
    for path in sorted(folder.glob("*_summary.csv")):
        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(f"{path}: expected exactly one row")

        row = df.iloc[0].to_dict()
        name = path.stem.replace("_summary", "")

        row["run"] = name
        row["model"] = model

        feature = str(row.get("feature_set", ""))

        if not feature:
            if "struct+cad" in name or "struct_cad" in name:
                feature = "struct_cad"
            elif "basic_cad" in name:
                feature = "basic_cad"
            elif "struct" in name:
                feature = "struct"
            elif "basic" in name:
                feature = "basic"

        row["feature_set"] = feature.replace(
            "struct+cad", "struct_cad"
        )

        exhaustive = row.get(
            "use_exhaustive",
            "_exh_" in name,
        )
        row["train_data"] = (
            "benchmark+exhaustive"
            if str(exhaustive).lower() in {"true", "1"}
            or "_exh_" in name
            else "benchmark"
        )

        direction = row.get(
            "aux_direction",
            "auxdir" in name,
        )
        row["aux_direction"] = (
            str(direction).lower() in {"true", "1"}
            or "auxdir" in name
        )

        loss_mask = row.get(
            "use_c_loss_mask",
            row.get("use_c_mask", "_cmask_" in name),
        )
        row["c_loss_mask"] = (
            str(loss_mask).lower() in {"true", "1"}
            or "_cmask_" in name
        )

        if (
            "test_cfilter_fp_reduced" not in row
            and "test_fp" in row
            and "test_cfilter_fp" in row
        ):
            row["test_cfilter_fp_reduced"] = (
                row["test_fp"] - row["test_cfilter_fp"]
            )

        rows.append(row)

results = pd.DataFrame(rows)

columns = [
    "run",
    "model",
    "feature_set",
    "train_data",
    "aux_direction",
    "c_loss_mask",
    "n_estimators",
    "seed",
    "best_epoch",
    "threshold",
    "test_ap",
    "test_mcc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_tp",
    "test_fp",
    "test_tn",
    "test_fn",
    "test_cfilter_fp",
    "test_cfilter_fp_reduced",
    "test_cfilter_mcc",
]

columns = [c for c in columns if c in results.columns]

results[columns].to_csv(OUTPUT, index=False)

print(results[columns].to_string(index=False))
print(f"\nWritten: {OUTPUT}")
