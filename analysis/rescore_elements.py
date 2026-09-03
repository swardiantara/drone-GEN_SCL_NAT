# -*- coding: utf-8 -*-
"""
Backfills performance_metrics.element_scores (source/eval_utils.py's
content-matched, multiset per-element ac/at/sp/ot scorer) into every
existing results-*.json under train_outputs/, in place, from each file's
already-saved per-example labels_pred/labels_correct -- no GPU, no
re-running inference.

This exists because compute_element_scores was added after the full grid
search had already been run: results-*.json files written before this
script only have the old, position-aligned ac_score/at_score/ot_score/
sp_score (see compute_element_scores's docstring in source/eval_utils.py
for why that's unreliable). Re-running this script after regenerating
results-*.json some other way is harmless -- it always recomputes
element_scores fresh from the file's stored examples and overwrites it.

Usage (from the repo root):
    python analysis/rescore_elements.py
    python analysis/rescore_elements.py --train_outputs train_outputs
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source'))
from eval_utils import compute_element_scores


def rescore_file(results_path):
    with open(results_path) as f:
        results = json.load(f)

    examples = results.get('examples')
    if not examples:
        print(f"[skip] {results_path}: no 'examples' field, nothing to rescore from")
        return False

    all_preds = [ex['labels_pred'] for ex in examples]
    all_labels = [ex['labels_correct'] for ex in examples]
    # labels are saved as JSON lists; compute_prf_averaged/_match_counts
    # need hashable (tuple) quads for the Counter-based multiset matching
    all_preds = [[tuple(quad) for quad in quads] for quads in all_preds]
    all_labels = [[tuple(quad) for quad in quads] for quads in all_labels]

    element_scores = compute_element_scores(all_preds, all_labels, silent=True)
    results.setdefault('performance_metrics', {})['element_scores'] = element_scores

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_outputs', default='train_outputs',
                         help="Root directory to scan for results-*.json files.")
    args = parser.parse_args()

    pattern = os.path.join(args.train_outputs, '**', 'results-*.json')
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print(f"No results-*.json files found under {args.train_outputs}/")
        return

    n_rescored = 0
    for path in paths:
        if rescore_file(path):
            n_rescored += 1
    print(f"Rescored {n_rescored}/{len(paths)} results-*.json file(s) under {args.train_outputs}/")


if __name__ == '__main__':
    main()
