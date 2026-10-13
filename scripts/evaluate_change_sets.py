#!/usr/bin/env python3
"""Evaluate exact change-set prediction per (graph_id, source, target).

Usage:
  python evaluate_change_sets.py results_gine results_rf --output-dir set_level_results
  python evaluate_change_sets.py path/to/run_predictions.csv --output-dir set_level_results

Uses stored prediction csv files.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['graph_id', 'source', 'target']
REQUIRED = KEY + ['argument', 'y_true', 'y_pred']


def read_one(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing required columns: {missing}')
    if df[KEY + ['argument']].isna().any().any():
        raise ValueError(f'{path}: missing update or argument identifiers')
    if df.duplicated(KEY + ['argument']).any():
        raise ValueError(f'{path}: duplicate argument within an update')
    for col in ['y_true', 'y_pred', 'y_pred_cfilter', 'in_c_delta']:
        if col in df and (df[col].isna().any() or not df[col].isin([0, 1]).all()):
            raise ValueError(f'{path}: {col} must contain only 0/1')
    if 'in_c_delta' in df and ((df.y_true == 1) & (df.in_c_delta == 0)).any():
        raise ValueError(f'{path}: positive ground truth outside candidate region')
    if 'y_pred_cfilter' in df and 'in_c_delta' in df:
        expected = df.y_pred.to_numpy(dtype=bool) & df.in_c_delta.to_numpy(dtype=bool)
        if not np.array_equal(expected, df.y_pred_cfilter.to_numpy(dtype=bool)):
            raise ValueError(f'{path}: filtered labels inconsistent with candidate mask')
    rows = []
    for identifiers, group in df.groupby(KEY, sort=False):
        actual = set(group.loc[group.y_true.eq(1), 'argument'])
        for variant, pred_col in [('unfiltered', 'y_pred'), ('filtered', 'y_pred_cfilter')]:
            if pred_col not in group:
                continue
            predicted = set(group.loc[group[pred_col].eq(1), 'argument'])
            union = actual | predicted
            intersection = actual & predicted
            rows.append(dict(zip(KEY, identifiers), variant=variant,
                             n_arguments=len(group), n_actual=len(actual),
                             n_predicted=len(predicted),
                             n_intersection=len(intersection),
                             exact_match=int(actual == predicted),
                             jaccard=len(intersection) / len(union) if union else 1.0,
                             actual_nonempty=int(bool(actual))))
    updates = pd.DataFrame(rows)
    summary = []
    for variant, g in updates.groupby('variant', sort=False):
        positive = g[g.actual_nonempty.eq(1)]
        empty = g[g.actual_nonempty.eq(0)]
        summary.append({
            'variant': variant,
            'n_updates': len(g),
            'n_nonempty_true_updates': len(positive),
            'n_empty_true_updates': len(empty),
            'exact_match': g.exact_match.mean(),
            'exact_match_nonempty': positive.exact_match.mean() if len(positive) else np.nan,
            'jaccard_mean': g.jaccard.mean(),
            'jaccard_mean_nonempty': positive.jaccard.mean() if len(positive) else np.nan,
            'exact_match_empty': empty.exact_match.mean() if len(empty) else np.nan,
        })
    return updates, pd.DataFrame(summary)


def discover(inputs):
    files = []
    for name in inputs:
        p = Path(name)
        if p.is_dir():
            files.extend(sorted(p.rglob('*_predictions.csv')))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return list(dict.fromkeys(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', help='Prediction CSV files or directories')
    parser.add_argument('--output-dir', type=Path, default=Path('set_level_results'))
    args = parser.parse_args()
    paths = discover(args.inputs)
    if not paths:
        parser.error('No *_predictions.csv files found')
    summaries = []
    baseline_truth = None
    base_keys = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        updates, summary = read_one(path)
        # Match update-level true-set sizes across files; never compare different test collections.
        truth = (updates.loc[updates.variant.eq('unfiltered'), KEY + ['n_arguments', 'n_actual']]
                 .sort_values(KEY).reset_index(drop=True))
        if baseline_truth is None:
            baseline_truth = truth
            base_keys = truth[KEY].copy()
        elif not truth.equals(baseline_truth):
            raise ValueError(f'{path}: mismatching test updates or ground truth')
        run = path.stem.removesuffix('_predictions')
        summary.insert(0, 'run', run)
        summary.insert(1, 'source_file', str(path))
        summaries.append(summary)
    result = pd.concat(summaries, ignore_index=True)
    # Trivial no-change classifier. Jaccard equals 1 for empty/empty and 0 for nonempty/empty.
    n = len(baseline_truth)
    empty = int(baseline_truth.n_actual.eq(0).sum())
    nonempty = n - empty
    baseline = pd.DataFrame([{
        'run': 'No Change', 'source_file': '', 'variant': 'unfiltered',
        'n_updates': n, 'n_nonempty_true_updates': nonempty,
        'n_empty_true_updates': empty, 'exact_match': empty/n,
        'exact_match_nonempty': 0.0 if nonempty else np.nan,
        'jaccard_mean': empty/n,
        'jaccard_mean_nonempty': 0.0 if nonempty else np.nan,
        'exact_match_empty': 1.0 if empty else np.nan,
    }])
    result = pd.concat([result, baseline], ignore_index=True)
    result.to_csv(args.output_dir/'set_level_results.csv', index=False)
    print(result[['run','variant','n_updates','n_nonempty_true_updates','exact_match',
                  'exact_match_nonempty','jaccard_mean','jaccard_mean_nonempty']].to_string(index=False))
    print('\nSaved:', args.output_dir/'set_level_results.csv')


if __name__ == '__main__':
    main()
