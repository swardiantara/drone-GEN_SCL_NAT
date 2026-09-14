# -*- coding: utf-8 -*-
"""
Case-study inference pipeline: runs the saved best-performing checkpoint
(configs/train_best_case_study_model.sh -> best-model/) over the seven
real-world flight-log evidence files under evidence/parsed/*.xlsx, one file
at a time (mirroring how a digital investigator works exhibit-by-exhibit,
never mixing predictions across cases), and writes two outputs per exhibit
under --output_dir (default case_study/predictions/seg-on or .../seg-off,
depending on --segment_messages -- see that flag below -- so a segmented and
an unsegmented run never overwrite each other's predictions):

1. <output_dir>/txt/<exhibit>.txt -- one line per evidence row,
   "<message>####<quads>", in the exact same format as data/<dataset>/*.txt
   (quads as [at, ac, sp, ot], ac/sp in their *raw* training label form, not
   the natural-language surface form the model actually generates -- see
   _to_raw_quad's docstring). Every row gets a line, even ones with zero
   predicted quads ("####[]"), so this file stays 1:1 with the evidence
   file's rows for manual correction.
2. <output_dir>/json/<exhibit>.json -- one entry per evidence row,
   {date, time, message_type, message} carried over unchanged plus a
   "quads" field (same [at, ac, sp, ot] raw-label lists as above), for
   building the forensic timeline.

Uses the same constrained-decoding vocabulary construction
(source/constrained_decoding.py) as training/evaluation, built from the
model's own training set (data/<dataset>/train.txt) -- not from the case
study data, which has no gold labels.

Optionally, --segment_messages splits each evidence message into individual
sentences with PySBD before prediction (the flight-log messages are often
several log lines joined with "; ", each really its own independent event --
see data/case_study/txt/*.txt), runs inference per sentence, then merges the
predicted quads for all of a message's sentences back into that message's
single row -- so the output stays 1:1 with the evidence file regardless of
whether segmentation is on.

Usage (from the repo root):
    python source/case_study_inference.py
    python source/case_study_inference.py --evidence_dir evidence/parsed --model_dir best-model
    python source/case_study_inference.py --use_cuda --batch_size 16
    python source/case_study_inference.py --segment_messages
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd
import pysbd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constrained_decoding import build_constrained_logits_processor, build_label_vocab
from eval_utils import extract_spans_para
from generate_data import dronesent2opinion, drone_full_mapping
from utils import load_mappings

MAPPINGS = load_mappings()

# reverse of the surface-form transforms applied when building training
# targets (source/data_utils.py get_para_asqp_targets / source/generate_data.py):
# ac/sp come back from the model in natural-language surface form (e.g. "the
# component", "ok"), but data/<dataset>/*.txt (and hence the case-study .txt
# output, which is meant to look like it) uses the raw training labels
# ("component", "positive"). at/ot need no reverse transform: extract_spans_para
# already turns a decoded "it" back into "NULL" for at, and ot's implicit
# value is the literal string "NULL" in both the surface and raw forms.
AC_SURFACE_TO_RAW = {surface: raw for surface, raw in drone_full_mapping}
SP_SURFACE_TO_RAW = {surface: raw for raw, surface in dronesent2opinion.items()}


def _to_raw_quad(quad, unmapped_counts):
    """
    quad: (ac_surface, at, sp_surface, ot) as returned by extract_spans_para.
    Returns [at, ac_raw, sp_raw, ot] -- the [at, ac, sp, ot] order and raw
    ac/sp label vocabulary used in data/<dataset>/*.txt. Falls back to the
    surface form (and increments unmapped_counts) if the model produced an
    ac/sp value outside the expected closed vocabulary -- constrained
    decoding makes this rare but not impossible (it restricts which tokens
    can be produced, not which whole phrases), and it's exactly the kind of
    row a human reviewer should double-check anyway.
    """
    ac_surface, at, sp_surface, ot = quad
    ac_raw = AC_SURFACE_TO_RAW.get(ac_surface)
    if ac_raw is None:
        unmapped_counts['ac'] += 1
        ac_raw = ac_surface
    sp_raw = SP_SURFACE_TO_RAW.get(sp_surface)
    if sp_raw is None:
        unmapped_counts['sp'] += 1
        sp_raw = sp_surface
    return [at, ac_raw, sp_raw, ot]


def get_aspect_category(dataset, task):
    """Same lookup as gen_scl_nat_main.py's get_aspect_category, without importing that (heavy) module."""
    domain = dataset.split('_')[1]
    acs = MAPPINGS[f'{domain}_full_mapping']
    return [ac[1] for ac in acs] if task == 'asqp' else [ac[0] for ac in acs]


def segment_message(segmenter, message):
    """Splits one message into sentences with PySBD. Falls back to the whole
    message as a single "sentence" if the segmenter finds nothing (e.g. a
    message with no sentence-ending punctuation).

    Evidence messages join separate log lines with "; " (see
    data/case_study/txt/*.txt); PySBD does not treat "; " as a boundary on
    its own, so it can leave a leading "; " on a segment that starts right
    after one -- strip that off so each segment reads like a standalone
    sentence before it's fed to the model."""
    raw_sentences = segmenter.segment(message)
    sentences = [s.strip().lstrip(';').strip() for s in raw_sentences]
    sentences = [s for s in sentences if s]
    return sentences or [message]


def load_evidence_file(path):
    df = pd.read_excel(path)
    required = {'date', 'time', 'message'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {sorted(missing)}")
    return df


def run_inference_on_messages(messages, model, tokenizer, device, args, category_vocab, sentiment_vocab):
    """Returns a list of raw-label quad lists (one per message), batched for speed."""
    all_quads = []
    unmapped_counts = {'ac': 0, 'sp': 0}
    extra_category_words = get_aspect_category(args.dataset, args.task) if args.constrained_decoding else None

    for start in range(0, len(messages), args.batch_size):
        batch_messages = messages[start:start + args.batch_size]

        tokenized = tokenizer.batch_encode_plus(
            batch_messages, max_length=args.max_seq_length, padding='max_length',
            truncation=True, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        gen_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                           max_length=args.max_seq_length * 2, num_beams=args.num_beams)

        if args.constrained_decoding:
            # decode the truncated source back to text, so the source-copy
            # vocabulary reflects exactly what the model saw (post-truncation),
            # same as evaluate()/evaluate_segmented() in gen_scl_nat_main.py
            source_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            logits_processor = build_constrained_logits_processor(
                tokenizer, args.task, source_texts, category_vocab, sentiment_vocab, args.num_beams,
                mappings=MAPPINGS, extra_category_words=extra_category_words)
            if logits_processor is not None:
                gen_kwargs['logits_processor'] = logits_processor

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        for text in decoded:
            quads = extract_spans_para(args.task, args.absa_task, text, 'pred')
            all_quads.append([_to_raw_quad(q, unmapped_counts) for q in quads])

    return all_quads, unmapped_counts


def process_evidence_file(path, model, tokenizer, device, args, category_vocab, sentiment_vocab, txt_dir, json_dir,
                           segmenter=None):
    exhibit_name = os.path.splitext(os.path.basename(path))[0]
    df = load_evidence_file(path)

    messages = [str(m).strip() if pd.notna(m) else '' for m in df['message']]
    empty_mask = [m == '' for m in messages]
    n_empty = sum(empty_mask)
    if n_empty:
        print(f"[{exhibit_name}] {n_empty} row(s) with an empty/NaN message -- predicting zero quads for them")

    non_empty_messages = [m for m, empty in zip(messages, empty_mask) if not empty]

    if segmenter is not None:
        # split each message into sentences, run inference over the flat
        # list of sentences, then merge each message's sentence-level quads
        # back into a single quad list for that message
        segments_per_message = [segment_message(segmenter, m) for m in non_empty_messages]
        flat_sentences = [s for segs in segments_per_message for s in segs]
        flat_quads, unmapped_counts = run_inference_on_messages(
            flat_sentences, model, tokenizer, device, args, category_vocab, sentiment_vocab)

        non_empty_quads = []
        it = iter(flat_quads)
        for segs in segments_per_message:
            merged = []
            for _ in segs:
                merged.extend(next(it))
            non_empty_quads.append(merged)
    else:
        non_empty_quads, unmapped_counts = run_inference_on_messages(
            non_empty_messages, model, tokenizer, device, args, category_vocab, sentiment_vocab)

    # re-interleave predictions back into the original row order, [] for empty rows
    quads_per_row = []
    it = iter(non_empty_quads)
    for empty in empty_mask:
        quads_per_row.append([] if empty else next(it))

    txt_path = os.path.join(txt_dir, f'{exhibit_name}.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for message, quads in zip(messages, quads_per_row):
            f.write(f'{message}####{quads}\n')

    json_entries = []
    for (_, row), quads in zip(df.iterrows(), quads_per_row):
        json_entries.append({
            'date': row['date'],
            'time': row['time'],
            'message_type': row.get('message_type'),
            'message': row['message'],
            'quads': quads,
        })
    json_path = os.path.join(json_dir, f'{exhibit_name}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_entries, f, indent=2, default=str)

    n_quads = sum(len(q) for q in quads_per_row)
    print(f"[{exhibit_name}] {len(df)} rows -> {n_quads} predicted quad(s) "
          f"({unmapped_counts['ac']} unmapped category, {unmapped_counts['sp']} unmapped sentiment) "
          f"-> {txt_path}, {json_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--evidence_dir', default='evidence/parsed',
                         help="Directory of evidence *.xlsx files (each with date/time/message_type/message columns).")
    parser.add_argument('--output_dir', default=None,
                         help="Root output directory; txt/ and json/ subfolders are created under it. Defaults "
                              "to case_study/predictions/seg-on or case_study/predictions/seg-off depending on "
                              "--segment_messages, so a seg-on and a seg-off run never overwrite each other's "
                              "predictions -- pass this explicitly to override.")
    parser.add_argument('--model_dir', default='best-model',
                         help="Directory with the saved HF checkpoint (configs/train_best_case_study_model.sh's output).")
    parser.add_argument('--dataset', default='acos_drone_binary',
                         help="Training dataset the checkpoint was trained on -- used to build the constrained-"
                              "decoding vocabulary from data/<dataset>/train.txt, and to look up the domain's "
                              "category list. Must match how best-model/ was trained.")
    parser.add_argument('--task', default='asqp', choices=['asqp', 'gen_scl_nat'])
    parser.add_argument('--absa_task', default='quad')
    parser.add_argument('--truncate', action='store_true',
                         help="Must match whether --truncate was passed when training best-model/.")
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16,
                         help="Internal batch size for generation within one evidence file (files are still "
                              "processed one at a time; this only affects throughput).")
    parser.add_argument('--disable_constrained_decoding', action='store_true',
                         help="Turn off constrained decoding. best-model/ was trained with it on, so leave "
                              "this off unless you specifically want to compare against unconstrained decoding.")
    parser.add_argument('--segment_messages', action='store_true',
                         help="Split each evidence message into sentences with PySBD before prediction, run "
                              "inference per sentence, then merge the predicted quads back into that message's "
                              "row. Off by default (whole message is fed to the model, matching how best-model/ "
                              "was trained).")
    parser.add_argument('--use_cuda', action='store_true', help="Force GPU. Auto-detected if not passed.")
    args = parser.parse_args()
    args.constrained_decoding = not args.disable_constrained_decoding
    if args.output_dir is None:
        args.output_dir = os.path.join('case_study', 'predictions',
                                        'seg-on' if args.segment_messages else 'seg-off')

    device = torch.device('cuda' if (args.use_cuda or torch.cuda.is_available()) else 'cpu')
    print(f"Device: {device}")

    print(f"Loading checkpoint from {args.model_dir}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    category_vocab, sentiment_vocab = None, None
    if args.constrained_decoding:
        category_vocab, sentiment_vocab = build_label_vocab(args.task, args.absa_task, args.dataset, args.truncate)
        print(f"Constrained-decoding vocab: {len(category_vocab)} categories, {len(sentiment_vocab)} sentiments "
              f"(from data/{args.dataset}/train.txt)")

    txt_dir = os.path.join(args.output_dir, 'txt')
    json_dir = os.path.join(args.output_dir, 'json')
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    segmenter = None
    if args.segment_messages:
        segmenter = pysbd.Segmenter(language='en', clean=False)
        print("Message segmentation: ON (PySBD)")

    evidence_paths = sorted(glob.glob(os.path.join(args.evidence_dir, '*.xlsx')))
    if not evidence_paths:
        print(f"No .xlsx files found under {args.evidence_dir}/")
        return

    for path in evidence_paths:
        process_evidence_file(path, model, tokenizer, device, args, category_vocab, sentiment_vocab, txt_dir, json_dir,
                               segmenter=segmenter)


if __name__ == '__main__':
    main()
