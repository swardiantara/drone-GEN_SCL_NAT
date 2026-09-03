# -*- coding: utf-8 -*-
"""
Shared utilities for the analysis/ scripts (recap.py, aggregate.py, statistics.py):
discovers completed grid-search runs under train_outputs/, reads their
args.json + results-*.json, and extracts config columns + per-metric-file
score columns as flat dicts, ready to load into a pandas DataFrame.

Scan scope: only the grid-search output layout produced by
configs/run_drone_paraphrase.sh / configs/run_drone_gen_scl_nat.sh via
source/gen_scl_nat_main.py --do_train --do_direct_eval, i.e.:
    train_outputs/<dataset>/<scenario>/<task>/<absa_task>/cont-{on,off}/cd-{on,off}/seg-{on,off}/<seed>/
(seed was originally the second-to-last path component, before cont-*/cd-*/
seg-*; it was moved to the end so every seed for one scenario sits side by
side on disk. find_run_dirs() matches both layouts.) Older/unrelated result
folders elsewhere under train_outputs/ (different layout) are ignored.

A run is only included if its args.json has a 'best_epoch' field -- i.e. it
was evaluated using source/gen_scl_nat_main.py's best-checkpoint-by-val-F1
selection. Runs from before that fix (no 'best_epoch') were evaluated on
whatever the last training epoch left in memory, which is not comparable to
the rest of the grid, so they're silently excluded here.
"""

import glob
import json
import os

# config columns shared by every recap/aggregate/statistics table, in the
# order they should appear as leading columns
CONFIG_COLUMNS = [
    'dataset', 'base_model', 'template', 'contrastive',
    'constrained_decoding', 'segmentation', 'seed',
]

TASK_TO_TEMPLATE = {
    'asqp': 'paraphrase',
    'gen_scl_nat': 'gen-scl-nat',
}


def _template_label(task):
    return TASK_TO_TEMPLATE.get(task, task)


def find_run_dirs(root='train_outputs'):
    """
    Returns sorted list of run directories matching the grid-search layout.
    Matches two shapes, since seed moved from just-before to just-after
    cont-*/cd-*/seg-*:
      new: .../cont-*/cd-*/seg-*/<seed>/args.json  (seg-*/<anything>/args.json)
      old: .../<seed>/[cont-*/]cd-*/seg-*/args.json  (seg-*/args.json directly)
    Matching on cd-*/seg-* (rather than a fixed total depth) keeps this
    working regardless of what other ablation levels sit above them.
    """
    patterns = [
        os.path.join(root, '**', 'cd-*', 'seg-*', 'args.json'),       # old: seed before cd-*/seg-*
        os.path.join(root, '**', 'cd-*', 'seg-*', '*', 'args.json'),  # new: seed after seg-*
    ]
    run_dirs = set()
    for pattern in patterns:
        run_dirs.update(os.path.dirname(p) for p in glob.glob(pattern, recursive=True))
    return sorted(run_dirs)


def load_run(run_dir):
    """
    Loads one run's args.json + results-*.json (picking the segmented vs.
    non-segmented filename based on args['use_segmentation']) and returns
    (config_dict, args_dict, results_dict), or None if the run should be
    skipped (missing best_epoch, or missing/unreadable results file).
    """
    args_path = os.path.join(run_dir, 'args.json')
    try:
        with open(args_path) as f:
            args = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] {run_dir}: could not read args.json ({exc})")
        return None

    if 'best_epoch' not in args or args.get('best_epoch') is None:
        print(f"[skip] {run_dir}: no best_epoch in args.json "
              f"(evaluated before best-checkpoint selection existed; not comparable)")
        return None

    dataset = args['dataset']
    result_filename = (f"results-{dataset}-segmented.json" if args.get('use_segmentation')
                        else f"results-{dataset}.json")
    results_path = os.path.join(run_dir, result_filename)
    try:
        with open(results_path) as f:
            results = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[skip] {run_dir}: could not read {result_filename} ({exc})")
        return None

    config = {
        'dataset': dataset,
        'base_model': args.get('scenario'),
        'template': _template_label(args.get('task')),
        'contrastive': 'yes' if float(args.get('cont_loss', 0.0)) > 0.0 else 'no',
        'constrained_decoding': 'yes' if args.get('constrained_decoding') else 'no',
        'segmentation': 'yes' if args.get('use_segmentation') else 'no',
        'seed': args.get('seed'),
    }
    return config, args, results


def iter_runs(root='train_outputs'):
    """Yields (run_dir, config, args, results) for every valid, included run."""
    for run_dir in find_run_dirs(root):
        loaded = load_run(run_dir)
        if loaded is None:
            continue
        config, args, results = loaded
        yield run_dir, config, args, results


def dget(d, *path, default=None):
    """Nested dict access: dget(d, 'a', 'b', 'c') == d['a']['b']['c'], or default if missing."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# --- per-metric-file column extractors --------------------------------------

def extract_set_multiset(results, kind):
    """kind: 'set' or 'multiset'. Returns the 9 P/R/F1 (micro/macro/weighted) columns."""
    node = dget(results, 'performance_metrics', 'set_multiset_scores', kind, default={})
    row = {}
    for agg in ('micro', 'macro', 'weighted'):
        agg_node = node.get(agg, {})
        for metric in ('precision', 'recall', 'f1'):
            row[f'{agg}_{metric}'] = agg_node.get(metric)
    return row


def extract_generative(results):
    """Flattens generative_score (BLEU/METEOR/ROUGE/WER) into one row of columns."""
    g = dget(results, 'generative_score', default={})
    row = {}

    bleu = g.get('bleu_score', {})
    row['bleu'] = bleu.get('bleu')
    row['bleu_brevity_penalty'] = bleu.get('brevity_penalty')
    row['bleu_length_ratio'] = bleu.get('length_ratio')
    row['bleu_reference_length'] = bleu.get('reference_length')
    row['bleu_translation_length'] = bleu.get('translation_length')
    for i, p in enumerate(bleu.get('precisions', []) or [], start=1):
        row[f'bleu_precision_{i}'] = p

    row['meteor'] = dget(g, 'meteor_score', 'meteor')

    rouge = g.get('rouge_score', {}) or {}
    for key in ('rouge1', 'rouge2', 'rougeL', 'rougeLsum'):
        row[key] = rouge.get(key)

    row['wer'] = g.get('wer_score')
    return row


def extract_element(results, element):
    """
    element: one of 'ac', 'at', 'ot', 'sp'. Returns the 9 P/R/F1
    (micro/macro/weighted) columns from source/eval_utils.py's
    compute_element_scores -- content-matched (multiset) per-element
    scoring, independent of full-quad positional alignment. See
    compute_element_scores's docstring for why this replaces the older
    position-aligned performance_metrics.{element}_score.
    """
    node = dget(results, 'performance_metrics', 'element_scores', element, default={})
    row = {}
    for agg in ('micro', 'macro', 'weighted'):
        agg_node = node.get(agg, {})
        for metric in ('precision', 'recall', 'f1'):
            row[f'{agg}_{metric}'] = agg_node.get(metric)
    return row


# metric-file name -> extractor function, used identically by recap.py and aggregate.py
METRIC_FILES = {
    'set-PRF': lambda results: extract_set_multiset(results, 'set'),
    'multiset-PRF': lambda results: extract_set_multiset(results, 'multiset'),
    'generative': extract_generative,
    'ac-PRF': lambda results: extract_element(results, 'ac'),
    'at-PRF': lambda results: extract_element(results, 'at'),
    'ot-PRF': lambda results: extract_element(results, 'ot'),
    'sp-PRF': lambda results: extract_element(results, 'sp'),
}


def build_metric_tables(root='train_outputs'):
    """
    Scans all valid runs once and returns {metric_file_name: [row_dict, ...]},
    where each row_dict is CONFIG_COLUMNS + that metric file's score columns.
    """
    tables = {name: [] for name in METRIC_FILES}
    n_runs = 0
    for run_dir, config, args, results in iter_runs(root):
        n_runs += 1
        for name, extractor in METRIC_FILES.items():
            row = dict(config)
            row.update(extractor(results))
            tables[name].append(row)
    print(f"Scanned {n_runs} valid run(s) under {root}/")
    return tables
