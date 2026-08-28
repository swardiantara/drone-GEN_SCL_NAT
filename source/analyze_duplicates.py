# -*- coding: utf-8 -*-
"""
CLI to identify duplicate quadruples within ASQP-style data splits (train/dev/test).

Motivation: if a non-trivial fraction of examples contain the same
(aspect, category, sentiment, opinion) quadruple more than once, set-based
evaluation is not appropriate (it silently drops repeats), and multiset
(bag) based evaluation should be used instead/in addition. This script
quantifies how common that situation is, per split, to justify that choice.

Run from the repository root, e.g.:
    python source/analyze_duplicates.py --dataset acos_drone_data
    python source/analyze_duplicates.py --dataset acos_drone_data --show_examples
    python source/analyze_duplicates.py --dataset acos_drone_data --output dup_report.json
"""

import argparse
import json
import os

from duplicate_utils import read_line_examples_from_file, analyze_split


def main():
    parser = argparse.ArgumentParser(description="Identify duplicate quadruples in ASQP-style data splits.")
    parser.add_argument('--dataset', type=str, required=True,
                         help="dataset directory name under data/, e.g. acos_drone_data")
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--output', type=str, default=None, help="optional path to dump a JSON report")
    parser.add_argument('--show_examples', action='store_true',
                         help="print the sentences that contain duplicate quads")
    args = parser.parse_args()

    report = {}
    for split in args.splits:
        data_path = f'data/{args.dataset}/{split}.txt'
        if not os.path.exists(data_path):
            print(f"[skip] {data_path} not found")
            continue

        sents, labels = read_line_examples_from_file(data_path)
        stats = analyze_split(sents, labels)
        report[split] = stats

        print(f"\n=== {args.dataset}/{split} ===")
        print(f"examples:                 {stats['n_examples']}")
        print(f"examples w/ duplicates:   {stats['n_examples_with_duplicates']} "
              f"({stats['pct_examples_with_duplicates']:.2f}%)")
        print(f"total gold quads:         {stats['n_total_quads']}")
        print(f"duplicate quad instances: {stats['n_duplicate_quad_instances']} "
              f"({stats['pct_duplicate_quad_instances']:.2f}%)")

        if args.show_examples:
            for ex in stats['examples']:
                print(f"  - {ex['sentence']}")
                for quad_str, count in ex['duplicate_quads'].items():
                    print(f"      x{count}: {quad_str}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to {args.output}")


if __name__ == '__main__':
    main()
