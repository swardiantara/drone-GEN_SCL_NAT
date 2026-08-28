# -*- coding: utf-8 -*-
"""
Utilities to detect duplicate (aspect, category, sentiment, opinion) quadruples
within individual ASQP-style examples.

This is used to justify multiset (bag) based evaluation over plain set-based
evaluation: when an example's gold label list contains the same quadruple
more than once, set-based scoring silently collapses those repeats to a
single occurrence (both `set(pred) & set(gold)` and `len(set(gold))` ignore
multiplicity), which under- or over-counts true positives/gold counts for
examples with repeated quadruples.

This module has no dependency on torch/transformers so it can be run as a
lightweight, standalone data-analysis step.
"""

from collections import Counter


def read_line_examples_from_file(data_path, silence=True):
    """
    Read data from file, each line formatted as: sent####labels
    Returns List[List[word]], List[List[quad]]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(eval(tuples))
    if not silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def find_duplicate_quads(label):
    """
    label: list of quads (each a list/tuple of 4 elements) for ONE example.
    Returns a dict {quad_tuple: count} restricted to quads occurring more than once.
    """
    counts = Counter(tuple(quad) for quad in label)
    return {quad: c for quad, c in counts.items() if c > 1}


def analyze_split(sents, labels):
    """
    Computes duplicate-quadruple statistics for one data split (train/dev/test).
    """
    n_examples = len(labels)
    n_examples_with_dupes = 0
    n_total_quads = 0
    n_duplicate_quad_instances = 0  # extra occurrences beyond the first, per quad
    per_example = []

    for sent, label in zip(sents, labels):
        n_total_quads += len(label)
        dupes = find_duplicate_quads(label)
        extra = sum(c - 1 for c in dupes.values())
        if dupes:
            n_examples_with_dupes += 1
            n_duplicate_quad_instances += extra
            per_example.append({
                'sentence': ' '.join(sent),
                'duplicate_quads': {' | '.join(map(str, q)): c for q, c in dupes.items()},
            })

    return {
        'n_examples': n_examples,
        'n_examples_with_duplicates': n_examples_with_dupes,
        'pct_examples_with_duplicates': (n_examples_with_dupes / n_examples * 100) if n_examples else 0.0,
        'n_total_quads': n_total_quads,
        'n_duplicate_quad_instances': n_duplicate_quad_instances,
        'pct_duplicate_quad_instances': (n_duplicate_quad_instances / n_total_quads * 100) if n_total_quads else 0.0,
        'examples': per_example,
    }


def analyze_file(data_path):
    sents, labels = read_line_examples_from_file(data_path)
    return analyze_split(sents, labels)
