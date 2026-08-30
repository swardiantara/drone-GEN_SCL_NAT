# -*- coding: utf-8 -*-
"""
Aggregates the per-run recap tables (see recap.py / common.py) across seeds:
for each metric file, groups by every config column except `seed`, and
computes the mean and std of each metric column over the seeds present for
that configuration. Writes the same 7 metric files under analysis/aggregate/.

Each output row is one configuration (dataset, base_model, template,
contrastive, constrained_decoding, segmentation) with, for every metric
column `m` from the corresponding recap table: `m_mean`, `m_std`, plus an
`n_seeds` column recording how many seeds contributed (so a configuration
run with fewer than the full 5 seeds -- e.g. a partially completed grid --
is visible rather than silently averaged over less data).

`std` uses ddof=1 (sample standard deviation); it is NaN when only one seed
is available for that configuration.

This reads directly from train_outputs/ (via common.py), the same source
recap.py uses, so it does not require recap.py to have been run first.

Usage (from the repo root):
    python analysis/aggregate.py
    python analysis/aggregate.py --train_outputs train_outputs --output_dir analysis/aggregate
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_metric_tables, CONFIG_COLUMNS

GROUP_COLUMNS = [c for c in CONFIG_COLUMNS if c != 'seed']


def aggregate_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    metric_columns = [c for c in df.columns if c not in CONFIG_COLUMNS]

    grouped = df.groupby(GROUP_COLUMNS, dropna=False)
    agg_df = grouped[metric_columns].agg(['mean', 'std']).reset_index()
    # flatten the ('metric', 'mean'/'std') MultiIndex columns to 'metric_mean'/'metric_std'
    agg_df.columns = [
        col[0] if col[1] == '' else f'{col[0]}_{col[1]}'
        for col in agg_df.columns
    ]

    n_seeds = grouped.size().reset_index(name='n_seeds')
    agg_df = agg_df.merge(n_seeds, on=GROUP_COLUMNS, how='left')

    # reorder: config columns, n_seeds, then metric_mean/metric_std pairs in original order
    ordered_metric_cols = [f'{m}_{stat}' for m in metric_columns for stat in ('mean', 'std')]
    agg_df = agg_df[GROUP_COLUMNS + ['n_seeds'] + ordered_metric_cols]
    agg_df = agg_df.sort_values(by=GROUP_COLUMNS).reset_index(drop=True)
    return agg_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_outputs', default='train_outputs',
                         help="Root directory to scan for grid-search run outputs.")
    parser.add_argument('--output_dir', default='analysis/aggregate',
                         help="Directory to write the per-metric .xlsx files into.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tables = build_metric_tables(args.train_outputs)

    for name, rows in tables.items():
        agg_df = aggregate_table(rows)
        out_path = os.path.join(args.output_dir, f'{name}.xlsx')
        agg_df.to_excel(out_path, index=False)
        print(f"Wrote {out_path} ({len(agg_df)} configurations, {len(agg_df.columns)} columns)")


if __name__ == '__main__':
    main()
