# -*- coding: utf-8 -*-
"""
Wilcoxon signed-rank test of every scenario against a configurable baseline,
on one main metric, computed per (dataset, base_model) group (so e.g. t5-base
scenarios are only ever compared against a t5-base baseline, never pooled
with flan-t5 results). Writes one Excel file under analysis/statistics/.

A "scenario" is one (template, contrastive, constrained_decoding,
segmentation) combination; the baseline is one specific such combination
(defaults to template=paraphrase, contrastive=yes, constrained_decoding=no,
segmentation=no -- override with the --baseline_* flags). The test pairs
runs by seed: only seeds present in both the baseline and the scenario are
used, so e.g. a scenario missing a seed the baseline has (or vice versa) is
still testable on the seeds they share, and is skipped with a note if they
share fewer than 2 seeds (Wilcoxon needs at least one non-zero paired
difference).

Reads directly from train_outputs/ (via common.py), the same source
recap.py/aggregate.py use.

Usage (from the repo root):
    python analysis/significance.py
    python analysis/significance.py --baseline_template paraphrase --baseline_contrastive yes \
        --baseline_constrained_decoding no --baseline_segmentation no
    python analysis/significance.py --metric set_micro_f1
"""

import argparse
import os
import sys

import numpy as np

import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import iter_runs, dget, CONFIG_COLUMNS

SCENARIO_COLUMNS = ['template', 'contrastive', 'constrained_decoding', 'segmentation']
GROUP_COLUMNS = ['dataset', 'base_model']

# name -> dotted path into a run's results.json, for --metric
METRIC_PATHS = {
    'multiset_micro_f1': ('performance_metrics', 'set_multiset_scores', 'multiset', 'micro', 'f1'),
    'multiset_macro_f1': ('performance_metrics', 'set_multiset_scores', 'multiset', 'macro', 'f1'),
    'multiset_weighted_f1': ('performance_metrics', 'set_multiset_scores', 'multiset', 'weighted', 'f1'),
    'set_micro_f1': ('performance_metrics', 'set_multiset_scores', 'set', 'micro', 'f1'),
    'set_macro_f1': ('performance_metrics', 'set_multiset_scores', 'set', 'macro', 'f1'),
    'set_weighted_f1': ('performance_metrics', 'set_multiset_scores', 'set', 'weighted', 'f1'),
    'quad_f1': ('performance_metrics', 'quad_score', 'f1_score'),
}


def load_metric_table(root, metric_path):
    rows = []
    for run_dir, config, args, results in iter_runs(root):
        value = dget(results, *metric_path)
        if value is None:
            print(f"[skip] {run_dir}: metric not found at {'.'.join(metric_path)}")
            continue
        row = dict(config)
        row['metric'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_wilcoxon(baseline_series, scenario_series):
    """
    baseline_series/scenario_series: pandas Series indexed by seed.
    Returns (n_pairs, baseline_mean, scenario_mean, statistic, p_value, note).
    """
    common_seeds = sorted(set(baseline_series.index) & set(scenario_series.index))
    n_pairs = len(common_seeds)
    if n_pairs == 0:
        return 0, None, None, None, None, 'no overlapping seeds with baseline'

    b = baseline_series.loc[common_seeds].to_numpy()
    s = scenario_series.loc[common_seeds].to_numpy()
    baseline_mean = float(b.mean())
    scenario_mean = float(s.mean())

    if n_pairs < 2:
        return n_pairs, baseline_mean, scenario_mean, None, None, 'fewer than 2 paired seeds; cannot run Wilcoxon'

    if np.allclose(b, s):
        # every paired difference is (numerically) zero -- scipy's default
        # zero_method drops these before ranking, which degenerates into a
        # divide-by-zero inside its normal-approximation path (RuntimeWarning,
        # though it still returns statistic=0/p=1.0). Short-circuit explicitly
        # instead of relying on that internal warning-producing edge case.
        return n_pairs, baseline_mean, scenario_mean, 0.0, 1.0, 'all paired differences are exactly zero'

    try:
        result = wilcoxon(b, s)
        return n_pairs, baseline_mean, scenario_mean, float(result.statistic), float(result.pvalue), ''
    except ValueError as exc:
        # e.g. all paired differences are exactly zero
        return n_pairs, baseline_mean, scenario_mean, None, None, f'wilcoxon failed: {exc}'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_outputs', default='train_outputs',
                         help="Root directory to scan for grid-search run outputs.")
    parser.add_argument('--output_dir', default='analysis/statistics',
                         help="Directory to write the Wilcoxon test .xlsx file into.")
    parser.add_argument('--metric', default='multiset_micro_f1', choices=sorted(METRIC_PATHS),
                         help="Which score to test.")
    parser.add_argument('--baseline_template', default='paraphrase', choices=['paraphrase', 'gen-scl-nat'])
    parser.add_argument('--baseline_contrastive', default='yes', choices=['yes', 'no'])
    parser.add_argument('--baseline_constrained_decoding', default='no', choices=['yes', 'no'])
    parser.add_argument('--baseline_segmentation', default='no', choices=['yes', 'no'])
    parser.add_argument('--alpha', type=float, default=0.05,
                         help="Significance threshold for the significant_at_alpha column.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metric_path = METRIC_PATHS[args.metric]
    df = load_metric_table(args.train_outputs, metric_path)
    if df.empty:
        print("No runs found; nothing to test.")
        return

    baseline_scenario = {
        'template': args.baseline_template,
        'contrastive': args.baseline_contrastive,
        'constrained_decoding': args.baseline_constrained_decoding,
        'segmentation': args.baseline_segmentation,
    }

    result_rows = []
    for group_key, group_df in df.groupby(GROUP_COLUMNS, dropna=False):
        dataset, base_model = group_key if isinstance(group_key, tuple) else (group_key,)

        baseline_mask = (group_df[SCENARIO_COLUMNS] == pd.Series(baseline_scenario)).all(axis=1)
        baseline_rows = group_df[baseline_mask]
        if baseline_rows.empty:
            print(f"[skip group] {dataset}/{base_model}: baseline scenario "
                  f"{baseline_scenario} not present in this group")
            continue
        baseline_series = baseline_rows.set_index('seed')['metric']

        scenarios = group_df[SCENARIO_COLUMNS].drop_duplicates()
        for _, scenario in scenarios.iterrows():
            scenario_dict = scenario.to_dict()
            if scenario_dict == baseline_scenario:
                continue  # don't test the baseline against itself

            scenario_mask = (group_df[SCENARIO_COLUMNS] == scenario).all(axis=1)
            scenario_rows = group_df[scenario_mask]
            scenario_series = scenario_rows.set_index('seed')['metric']

            n_pairs, baseline_mean, scenario_mean, statistic, p_value, note = run_wilcoxon(
                baseline_series, scenario_series)

            result_rows.append({
                'dataset': dataset,
                'base_model': base_model,
                'metric': args.metric,
                'baseline_template': baseline_scenario['template'],
                'baseline_contrastive': baseline_scenario['contrastive'],
                'baseline_constrained_decoding': baseline_scenario['constrained_decoding'],
                'baseline_segmentation': baseline_scenario['segmentation'],
                'scenario_template': scenario_dict['template'],
                'scenario_contrastive': scenario_dict['contrastive'],
                'scenario_constrained_decoding': scenario_dict['constrained_decoding'],
                'scenario_segmentation': scenario_dict['segmentation'],
                'n_pairs': n_pairs,
                'baseline_mean': baseline_mean,
                'scenario_mean': scenario_mean,
                'mean_diff_scenario_minus_baseline':
                    (scenario_mean - baseline_mean) if baseline_mean is not None and scenario_mean is not None else None,
                'wilcoxon_statistic': statistic,
                'p_value': p_value,
                f'significant_at_{args.alpha}': (p_value < args.alpha) if p_value is not None else None,
                'note': note,
            })

    result_df = pd.DataFrame(result_rows)
    out_path = os.path.join(args.output_dir, f'wilcoxon_{args.metric}.xlsx')
    result_df.to_excel(out_path, index=False)
    print(f"Wrote {out_path} ({len(result_df)} comparisons)")


if __name__ == '__main__':
    main()
