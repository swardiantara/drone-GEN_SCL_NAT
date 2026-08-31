# -*- coding: utf-8 -*-
"""
Recaps dataset statistics for every dataset under data/ (or a chosen subset)
into a multi-sheet Excel file: number of messages, number of quads, number
of implicit aspects/opinions, sentiment label distribution, and aspect
category distribution -- per (dataset, split).

Quad format (as read from each `sent####[[at, ac, sentiment, ot], ...]`
line): [aspect_term, category, sentiment, opinion_term]. An aspect/opinion
is "implicit" when its term is the literal string 'NULL' (see
source/generate_data.py's ex_contains_implicit_aspect/_opinion, whose
definitions this mirrors).

This only reads data/<dataset>/{train,dev,test}.txt via
source/duplicate_utils.py's standalone reader, so it has no torch/
transformers dependency.

Output: analysis/dataset_statistics.xlsx, with sheets:
    overview                     one row per (dataset, split)
    sentiment_distribution       one row per (dataset, split, sentiment label)
    aspect_category_distribution one row per (dataset, split, category)

Usage (from the repo root):
    python analysis/dataset_statistics.py
    python analysis/dataset_statistics.py --datasets acos_drone_binary acos_drone_multi
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'source'))

from duplicate_utils import read_line_examples_from_file  # noqa: E402

SPLITS = ('train', 'dev', 'test')


def discover_datasets(data_root):
    datasets = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and all(os.path.exists(os.path.join(path, f'{s}.txt')) for s in SPLITS):
            datasets.append(name)
    return datasets


def analyze_split(dataset, split, data_root):
    data_path = os.path.join(data_root, dataset, f'{split}.txt')
    if not os.path.exists(data_path):
        return None

    sents, labels = read_line_examples_from_file(data_path)

    n_messages = len(sents)
    n_quads = sum(len(ex) for ex in labels)

    n_implicit_aspect = 0
    n_implicit_opinion = 0
    n_full_implicit = 0
    sentiment_counter = Counter()
    category_counter = Counter()

    for ex in labels:
        for quad in ex:
            at, ac, sp, ot = quad[0], quad[1], quad[2], quad[3]
            aspect_implicit = (at == 'NULL')
            opinion_implicit = (ot == 'NULL')
            if aspect_implicit:
                n_implicit_aspect += 1
            if opinion_implicit:
                n_implicit_opinion += 1
            if aspect_implicit and opinion_implicit:
                n_full_implicit += 1
            sentiment_counter[sp] += 1
            category_counter[ac] += 1

    overview = {
        'dataset': dataset,
        'split': split,
        'n_messages': n_messages,
        'n_quads': n_quads,
        'avg_quads_per_message': (n_quads / n_messages) if n_messages else 0.0,
        'n_implicit_aspect': n_implicit_aspect,
        'pct_implicit_aspect': (n_implicit_aspect / n_quads * 100) if n_quads else 0.0,
        'n_implicit_opinion': n_implicit_opinion,
        'pct_implicit_opinion': (n_implicit_opinion / n_quads * 100) if n_quads else 0.0,
        'n_full_implicit': n_full_implicit,
        'pct_full_implicit': (n_full_implicit / n_quads * 100) if n_quads else 0.0,
        'n_unique_sentiment_labels': len(sentiment_counter),
        'n_unique_categories': len(category_counter),
    }
    return overview, sentiment_counter, category_counter, n_quads


def build_distribution_rows(dataset, split, counter, n_quads, label_column):
    rows = []
    for label, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append({
            'dataset': dataset,
            'split': split,
            label_column: label,
            'count': count,
            'pct_of_quads': (count / n_quads * 100) if n_quads else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', default='data',
                         help="Root directory containing one subdirectory per dataset.")
    parser.add_argument('--datasets', nargs='+', default=None,
                         help="Dataset directory names to include. Default: auto-discover every "
                              "subdirectory of --data_root that has train/dev/test.txt.")
    parser.add_argument('--output', default='analysis/dataset_statistics.xlsx')
    args = parser.parse_args()

    datasets = args.datasets or discover_datasets(args.data_root)
    if not datasets:
        print(f"No datasets found under {args.data_root}/")
        return
    print(f"Datasets: {', '.join(datasets)}")

    overview_rows = []
    sentiment_rows = []
    category_rows = []

    for dataset in datasets:
        for split in SPLITS:
            result = analyze_split(dataset, split, args.data_root)
            if result is None:
                print(f"[skip] data/{dataset}/{split}.txt not found")
                continue
            overview, sentiment_counter, category_counter, n_quads = result
            overview_rows.append(overview)
            sentiment_rows.extend(build_distribution_rows(
                dataset, split, sentiment_counter, n_quads, 'sentiment'))
            category_rows.extend(build_distribution_rows(
                dataset, split, category_counter, n_quads, 'category'))

    overview_df = pd.DataFrame(overview_rows)
    sentiment_df = pd.DataFrame(sentiment_rows)
    category_df = pd.DataFrame(category_rows)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with pd.ExcelWriter(args.output, engine='openpyxl') as writer:
        overview_df.to_excel(writer, sheet_name='overview', index=False)
        sentiment_df.to_excel(writer, sheet_name='sentiment_distribution', index=False)
        category_df.to_excel(writer, sheet_name='aspect_category_distribution', index=False)

    print(f"Wrote {args.output} "
          f"({len(overview_df)} dataset/split rows, "
          f"{len(sentiment_df)} sentiment rows, "
          f"{len(category_df)} category rows)")


if __name__ == '__main__':
    main()
