Code and data accompanying a thesis on predicting which arguments change their credulous acceptance status under admissible semantics after adding one attack to a Dung argumentation framework (AF). The repository includes a labelled benchmark, a small exhaustive study, GINE and Random Forest models, and evaluation scripts.

## Data

- Prediction benchmark: 208 base AFs (135 training / 21 validation / 52 test), with 2,070 sampled single-attack additions. `instances.csv` contains the attack endpoints and exact change, gain, and loss labels.
- Exhaustive study: 36 small AFs and 43,982 evaluated attack additions, including self-attacks.
- Results: `all_results.csv` and `set_level_results.csv` contain the reported aggregate and set-level evaluation results.

The benchmark was constructed to cover different AF structures and update situations, no representative sample

```text
scripts/                   Python scripts (see overview below)
dataset/
  settings.csv             Recorded sampling settings
  split_graphs.csv         AF metadata and data split
  test_graphs.csv          Test-AF properties for graph-wise analysis
  train/                   Training instances.csv, updates.csv, base_graphs/, base_cad/
  val/                     Validation instances.csv, updates.csv, base_graphs/, base_cad/
  test/                    Test instances.csv, updates.csv, base_graphs/, base_cad/
ds_small/
  updates.csv              Exact exhaustive-update labels
  train/                   instances.csv, base_graphs/, base_cad/
  analysis_exh_small/      AF metadata and selected summary tables
requirements.txt
```

`dataset/` and `ds_small/` contain the **precomputed benchmark**, not directories that must first be generated to run the models. In each split, `instances.csv` lists the labelled attack additions; `base_graphs/` contains the initial AFs, and `base_cad/` contains the set of initial credulously accepted args (`<file_name>.dca.txt`). The AF files are in APX or TGF format. The model scripts read these files directly, so dataset generation and an exact solver are not required for training.

`prepare_dataset.py` created the split directories during the original benchmark construction.

## Install and run a model

Work from the repository root.

The training programs use **settings at the top of each `.py` file**, not command-line flags. The options relevant to the experiments are:

| Script | Settings | Meaning |
| --- | --- | --- |
| `train_nn_gine.py` | `FEATURE_SET` | `basic`, `basic_cad`, `struct`, or `struct_cad` |
| `train_nn_gine.py` | `USE_EXHAUSTIVE`, `USE_C_MASK` | Add small-AF training data / restrict training loss to the candidate region |
| `train_nn_gine_ch_dir.py` | Same settings, plus `AUX_WEIGHT` | Train GINE with an auxiliary gain/loss objective |
| `train_rf.py` | `USE_CAD`, `USE_EXHAUSTIVE` | Use initial acceptance status / add small-AF training data |

Set `USE_EXHAUSTIVE=False` to train only on the main benchmark; the other setting requires the 36 small AFs. For a standard unmasked GINE experiment, set `USE_C_MASK=False`. Paths are defined by `DATASET_DIR`, `SMALL_DIR`, and `OUTPUT_DIR` in the training scripts. To train the other models:

Each run writes prediction and summary CSVs to `results_gine/` or `results_rf/`, along with optional model checkpoints. Run the desired configurations separately. Repeating the same configuration may overwrite its previous output files.

## Evaluate results

```bash
python scripts/collect.py
python scripts/make_result_tables_final.py
python scripts/evaluate_change_sets.py results_gine results_rf --output-dir set_level_results
```

**Note:** `collect.py` reads all available training summaries and overwrites `all_results.csv`. Use it only when the intended runs are present. The set-level evaluation requires the generated per-argument `*_predictions.csv` files, not just `all_results.csv`.

For analysis specify matching prediction files:

```bash
python scripts/evaluate_graph_properties.py \
  --struct results_gine/STRUCT_PREDICTIONS.csv \
  --struct_cad results_gine/STRUCT_CAD_PREDICTIONS.csv \
  --graphs dataset/test_graphs.csv \
  --output-dir graph_diagnostics
```

Replace the uppercase filenames with the actual generated prediction CSV filenames. Run separately for the desired training conditions.

With the 36 AF files and their initial acceptance files in `ds_small/train/`, the labels can be reanalysed:

```bash
python scripts/analyze_exhaustive.py \
  --pool ds_small/analysis_exh_small/exhaustive_graphs.csv \
  --updates ds_small/updates.csv \
  --graphs-dir ds_small/train/base_graphs \
  --base-cad-dir ds_small/train/base_cad \
  --output-dir work/exhaustive_analysis
```

## Script overview

| Stage               | Scripts and purpose                                                                                                                                                                                                                            |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scan and select AFs | `scan_graphs.py` (read AFs and descriptors), `build_benchmark_pool.py` (combine candidate pools), `select_prepool.py` and `select_final_pool.py` (select structurally diverse AFs)                                                             |
| Construct dataset   | `prepare_attack_candidates.py` (update classes), `prepare_dataset.py` (split, sample, label), `compute_exhaustive_updates.py` (evaluate small AFs)                                                                                             |
| Train               | `train_nn_gine.py` (GINE), `train_nn_gine_ch_dir.py` (GINE + direction supervision), `train_rf.py` (Random Forest)                                                                                                                             |
| Evaluate            | `collect.py` (combine summaries), `make_result_tables_final.py` (result tables), `evaluate_change_sets.py` (exact-set match), `evaluate_graph_properties.py` (graph-property diagnostics), `analyze_exhaustive_compact_v2.py` (exhaustive analysis) |
| Shared              | `parse_graphs.py`, `cad.py`, `attack_properties.py`, `prepare_pyg_data.py`, `evaluation.py` (imported by other scripts; not separate command-line programs)                                                                                    |

The scripts for creating or analysing datasets accept command-line parameters.

```bash
python scripts/prepare_dataset.py --help
python scripts/compute_exhaustive_updates.py --help
```

Optional: create a benchmark set from raw graph pools
```bash
python scripts/scan_graphs.py --input-dir /path/to/raw-afs --output work/scanned.csv --max-args 700 --max-attacks 30000
python scripts/build_benchmark_pool.py --pools work/scanned.csv --output work/pool.csv
python scripts/select_prepool.py --input work/pool.csv --output work/prepool.csv --solver /path/to/solver --acceptance-dir work/base_cad --n-select 700 --timeout 60
python scripts/select_final_pool.py --input work/prepool.csv --output work/selected180.csv --n-select 180 --low-acceptance-threshold 0.2 --max-low-acceptance-fraction 0.2
```
prepare attack sampling and construct dataset
```bash
python scripts/prepare_attack_candidates.py --pool work/final208.csv --root . --acceptance-dir work/base_cad --output-dir work/candidates
python scripts/prepare_dataset.py --pool work/final208.csv --candidates-dir work/candidates --base-cad-dir work/base_cad --root . --output-dir work/new_dataset --seed 42 --solver /path/to/solver --timeout 60
```

To regenerate the small-AF labels from their original graphs:

```bash
python scripts/compute_exhaustive_updates.py --pool ds_small/analysis_exh_small/exhaustive_graphs.csv --base-cad-dir ds_small/train/base_cad --root . --output-dir work/recomputed_small --solver /path/to/solver --timeout 60
```


## Dependencies and provenance
`cad.py` expects an external argumentation solver. Note that the scripts use .tgf and .apx but no .af files 
benchmarks were constructed using available benchmark generators, some benchmarks are taken form the ICCMA benchmark sets https://argumentationcompetition.org/
