# -*- coding: utf-8 -*-
"""
Diagnoses the ADFLER-based segmentation used by --use_segmentation
(source/segmentation_utils.py): runs the segmentation model on a dataset
split's raw messages and dumps the *token-level* predictions -- not just the
final Event-sentence list that SentenceSegmenter.segment() returns -- so you
can see exactly what the model tagged for every message: the raw BIOES tags,
the boundaries extract_boundaries_with_types() derives from them (with their
Event/NonEvent type), and whether the "no Event segment found" fallback to
the whole message was triggered.

This directly reuses source/segmentation_utils.py's SentenceSegmenter
(specifically its internal _predict_tags helper, to get at the raw per-token
predictions it normally discards after computing segments) and
extract_boundaries_with_types, so the diagnostic reflects exactly the same
code path --use_segmentation exercises during evaluation.

Output: one JSON file under analysis/segmentation/, with a summary block
(tag distribution, how many messages got zero boundaries, how many got more
than one Event segment) and one record per message.

Usage (from the repo root):
    python analysis/diagnose_segmentation.py
    python analysis/diagnose_segmentation.py --dataset acos_drone_binary --split test
    python analysis/diagnose_segmentation.py --segmentation_use_cuda
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'source'))

from segmentation_utils import SentenceSegmenter, extract_boundaries_with_types, DEFAULT_MODEL_DIR  # noqa: E402


def read_messages(dataset, split):
    data_path = os.path.join(_REPO_ROOT, 'data', dataset, f'{split}.txt')
    sents = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words, _ = line.split('####')
            sents.append(words.split())
    return sents


def diagnose(messages, segmenter):
    words_per_message, tags_per_message = segmenter._predict_tags(messages)  # noqa: SLF001 (intentional reuse)

    records = []
    n_zero_boundaries = 0
    n_multi_event = 0
    n_used_fallback = 0
    tag_counts = {}

    for idx, (message, words, tags) in enumerate(zip(messages, words_per_message, tags_per_message)):
        boundaries = extract_boundaries_with_types(tags)
        segments = [
            {
                'start': start,
                'end': end,
                'entity_type': entity_type,
                'text': ' '.join(words[start:end + 1]),
            }
            for start, end, entity_type in boundaries
        ]
        n_event = sum(1 for s in segments if s['entity_type'] == SentenceSegmenter.EVENT_TYPE)
        used_fallback = n_event == 0

        if not boundaries:
            n_zero_boundaries += 1
        if n_event > 1:
            n_multi_event += 1
        if used_fallback:
            n_used_fallback += 1
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

        records.append({
            'index': idx,
            'message': message,
            'n_tokens': len(words),
            'token_tag_pairs': list(zip(words, tags)),
            'boundaries': segments,
            'n_boundaries': len(segments),
            'n_event_segments': n_event,
            'used_fallback_whole_message': used_fallback,
        })

    summary = {
        'n_messages': len(messages),
        'n_messages_with_zero_boundaries': n_zero_boundaries,
        'n_messages_with_multiple_event_segments': n_multi_event,
        'n_messages_using_whole_message_fallback': n_used_fallback,
        'tag_distribution': dict(sorted(tag_counts.items())),
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='acos_drone_binary',
                         help="Dataset directory under data/ to read messages from.")
    parser.add_argument('--split', default='test', choices=['train', 'dev', 'test'])
    parser.add_argument('--segmentation_model_dir', default=DEFAULT_MODEL_DIR,
                         help="Path or Hugging Face Hub id of the ADFLER-style checkpoint.")
    parser.add_argument('--segmentation_model_type', default='bert')
    parser.add_argument('--segmentation_use_cuda', action='store_true')
    parser.add_argument('--output', default=None,
                         help="Output JSON path. Defaults to "
                              "analysis/segmentation/<dataset>_<split>_diagnostics.json")
    args = parser.parse_args()

    messages_words = read_messages(args.dataset, args.split)
    messages = [' '.join(words) for words in messages_words]
    print(f"Loaded {len(messages)} messages from data/{args.dataset}/{args.split}.txt")

    print(f"Loading segmentation model {args.segmentation_model_dir} "
          f"(model_type={args.segmentation_model_type}, use_cuda={args.segmentation_use_cuda}) ...")
    segmenter = SentenceSegmenter(args.segmentation_model_dir, use_cuda=args.segmentation_use_cuda,
                                   model_type=args.segmentation_model_type)

    summary, records = diagnose(messages, segmenter)
    summary.update({
        'dataset': args.dataset,
        'split': args.split,
        'segmentation_model_dir': args.segmentation_model_dir,
        'segmentation_model_type': args.segmentation_model_type,
    })

    print(json.dumps(summary, indent=2))

    output_path = args.output or os.path.join(
        'analysis', 'segmentation', f'{args.dataset}_{args.split}_diagnostics.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'summary': summary, 'records': records}, f, indent=2)
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    main()
