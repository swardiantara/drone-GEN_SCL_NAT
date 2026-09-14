# -*- coding: utf-8 -*-
"""
Scores the case-study predictions (source/case_study_inference.py's output
under case_study/predictions/{seg-off,seg-on}/txt/<flight_log>.txt) against
the case-study gold labels under data/case_study/txt/<flight_log>.txt, one
flight log at a time.

Both the gold and predicted .txt files are in the same
"<message>####[[at, ac, sp, ot], ...]" format as data/<dataset>/*.txt (see
source/duplicate_utils.py's read_line_examples_from_file), already fully
parsed into quads -- unlike the grid-search results-*.json files, there is no
raw generated text to run source/eval_utils.py's extract_spans_para over, so
this reuses eval_utils's scoring functions (compute_f1_scores,
compute_scores_from_quads, compute_element_scores) directly on the quad
lists, after reordering each quad from the case-study [at, ac, sp, ot] label
order to the (ac, at, sp, ot) tuple order those functions expect (see
eval_utils.py's ELEMENT_INDEX comment).

For each flight log, writes analysis/case_study/<seg>/<flight_log>.json with
the same 'performance_metrics' shape gen_scl_nat_main.py's evaluate() writes
(position-aligned + set/multiset + element scores), plus an 'examples' list
(message/gold quads/predicted quads per line) for manual error inspection.

Usage (from the repo root):
    python analysis/evaluate_case_study.py
    python analysis/evaluate_case_study.py --predictions_dir case_study/predictions --gold_dir data/case_study/txt
"""

import argparse
import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'source'))

from duplicate_utils import read_line_examples_from_file  # noqa: E402
from eval_utils import (  # noqa: E402
    compute_f1_scores, compute_scores_from_quads, compute_element_scores)

SEGMENTATION_SETTINGS = ('seg-off', 'seg-on')


def to_ac_at_sp_ot(quads):
    """Case-study .txt quads are [at, ac, sp, ot] (same order as
    data/<dataset>/*.txt); eval_utils's scoring functions expect
    (ac, at, sp, ot) tuples (its own ELEMENT_INDEX/extract_spans_para
    convention)."""
    return [(ac, at, sp, ot) for at, ac, sp, ot in quads]


def discover_flight_logs(gold_dir):
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(gold_dir, '*.txt'))
    )


def evaluate_flight_log(gold_path, pred_path):
    gold_sents, gold_labels = read_line_examples_from_file(gold_path)
    pred_sents, pred_labels = read_line_examples_from_file(pred_path)

    if len(gold_sents) != len(pred_sents):
        raise ValueError(f"{gold_path} has {len(gold_sents)} line(s) but {pred_path} has "
                          f"{len(pred_sents)} -- predictions must be line-for-line aligned with gold "
                          "(re-run source/case_study_inference.py over the matching evidence file).")

    all_labels = [to_ac_at_sp_ot(quads) for quads in gold_labels]
    all_preds = [to_ac_at_sp_ot(quads) for quads in pred_labels]

    scores = compute_f1_scores(all_preds, all_labels, silent=True)
    scores['set_multiset_scores'] = compute_scores_from_quads(all_preds, all_labels, silent=True)
    scores['element_scores'] = compute_element_scores(all_preds, all_labels, silent=True)

    examples = []
    for gold_sent, gold_quads, pred_quads in zip(gold_sents, gold_labels, pred_labels):
        examples.append({
            'message': ' '.join(gold_sent),
            'labels_correct': gold_quads,
            'labels_pred': pred_quads,
        })

    return {'performance_metrics': scores, 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--predictions_dir', default='case_study/predictions',
                         help="Root with one subfolder per segmentation setting (seg-off/, seg-on/), each "
                              "holding a txt/<flight_log>.txt per source/case_study_inference.py.")
    parser.add_argument('--gold_dir', default='data/case_study/txt',
                         help="Directory of gold <flight_log>.txt files.")
    parser.add_argument('--output_dir', default='analysis/case_study',
                         help="Root output directory; one subfolder per segmentation setting is created under it.")
    args = parser.parse_args()

    flight_logs = discover_flight_logs(args.gold_dir)
    if not flight_logs:
        print(f"No gold flight logs found under {args.gold_dir}/")
        return

    for seg in SEGMENTATION_SETTINGS:
        pred_txt_dir = os.path.join(args.predictions_dir, seg, 'txt')
        if not os.path.isdir(pred_txt_dir):
            print(f"[skip] {pred_txt_dir}/ not found")
            continue

        out_dir = os.path.join(args.output_dir, seg)
        os.makedirs(out_dir, exist_ok=True)

        for flight_log in flight_logs:
            gold_path = os.path.join(args.gold_dir, f'{flight_log}.txt')
            pred_path = os.path.join(pred_txt_dir, f'{flight_log}.txt')
            if not os.path.exists(pred_path):
                print(f"[skip] {pred_path} not found")
                continue

            result = evaluate_flight_log(gold_path, pred_path)

            out_path = os.path.join(out_dir, f'{flight_log}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            quad_f1 = result['performance_metrics']['quad_score']['f1_score']
            print(f"[{seg}/{flight_log}] quad F1={quad_f1:.4f} -> {out_path}")


if __name__ == '__main__':
    main()
