# -*- coding: utf-8 -*-
"""
Recaps all completed grid-search experiments (see common.py for scan scope)
into one Excel file per metric under analysis/recap/:

    set-PRF.xlsx       set-based precision/recall/F1, micro/macro/weighted
    multiset-PRF.xlsx  multiset (bag) precision/recall/F1, micro/macro/weighted
    generative.xlsx    BLEU / METEOR / ROUGE / WER, all sub-metrics
    ac-PRF.xlsx         aspect-category element-level precision/recall/F1
    at-PRF.xlsx         aspect-term element-level precision/recall/F1
    ot-PRF.xlsx         opinion-term element-level precision/recall/F1
    sp-PRF.xlsx         sentiment-polarity element-level precision/recall/F1

Each row is one run: config columns (dataset, base_model, template,
contrastive, constrained_decoding, segmentation, seed) + that file's metric
columns.

Usage (from the repo root):
    python analysis/recap.py
    python analysis/recap.py --train_outputs train_outputs --output_dir analysis/recap
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_metric_tables, CONFIG_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_outputs', default='train_outputs',
                         help="Root directory to scan for grid-search run outputs.")
    parser.add_argument('--output_dir', default='analysis/recap',
                         help="Directory to write the per-metric .xlsx files into.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tables = build_metric_tables(args.train_outputs)

    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by=CONFIG_COLUMNS).reset_index(drop=True)
        out_path = os.path.join(args.output_dir, f'{name}.xlsx')
        df.to_excel(out_path, index=False)
        print(f"Wrote {out_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == '__main__':
    main()
