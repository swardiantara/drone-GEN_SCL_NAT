# -*- coding: utf-8 -*-
"""
Optional constrained decoding for ASQP-style generation.

Design: this implements *vocabulary-restricted* constrained decoding, not a
full grammar/state-machine decoder. At every generation step, the set of
allowed next-token ids for a given (source) example is the union of:
  - token ids that appear in that example's source input (so the model can
    only "copy" aspect/opinion spans that are actually present in the input),
  - token ids belonging to the closed label vocabularies (categories,
    sentiment polarities) observed in the training target sequences,
  - the fixed template/structural tokens used by the task's output format
    (e.g. "is", "because", "|", "THE", "IS", "[SSEP]"),
  - special tokens needed to end generation (eos/pad/unk).

This is a lightweight, self-consistent approximation (it restricts *which*
tokens can be produced, without enforcing the exact template grammar/order)
that is cheap to compute and works with HuggingFace's `generate()` via a
`LogitsProcessor`, which is what we need to run an ablation on the effect of
constraining decoding to the input+label vocabulary vs. free generation.
"""

import string

from transformers import LogitsProcessor, LogitsProcessorList

# fixed structural/template tokens per task, used verbatim by the target
# formats built in data_utils.py / generate_data.py
TASK_TEMPLATE_WORDS = {
    'asqp': ['is', 'because', '[SSEP]', 'it'],
    'gen_scl_nat': ['|', 'THE', 'IS', '[SSEP]', 'it'],
    'gen_scl_nat_wo_intra': ['is', 'because', '[SSEP]', 'it'],
    'gen_scl_nat_wo_nat_cats': ['|', 'THE', 'IS', '[SSEP]', 'it'],
    'gen_scl_nat_wo_sorting': ['|', 'THE', 'IS', '[SSEP]', 'it'],
}


def build_label_vocab(task, dataset_dir, truncate=False):
    """
    Empirically derives the closed category/sentiment surface-form vocabulary
    from the *training* target sequences, by reusing the exact same
    generation (data_utils.get_transformed_io) and parsing (eval_utils.
    extract_spans_para) logic used elsewhere in the pipeline. This keeps the
    constrained-decoding vocabulary self-consistent with whatever surface
    forms the model is actually trained to produce for a given task/dataset,
    without having to special-case category-name mappings per domain.
    """
    from data_utils import get_transformed_io
    from eval_utils import extract_spans_para

    _, targets, _ = get_transformed_io(f'data/{dataset_dir}/train.txt', dataset_dir, task, 'train', truncate)
    target_texts = [t if isinstance(t, str) else " ".join(t) for t in targets]

    categories, sentiments = set(), set()
    for text in target_texts:
        for ac, at, sp, ot in extract_spans_para(task, text, 'gold'):
            if ac:
                categories.add(ac)
            if sp:
                sentiments.add(sp)

    return categories, sentiments


def _token_ids_for_words(tokenizer, words):
    ids = set()
    for w in words:
        if not w:
            continue
        ids.update(tokenizer.encode(w, add_special_tokens=False))
        ids.update(tokenizer.encode(' ' + w, add_special_tokens=False))
    return ids


class ConstrainedVocabProcessor(LogitsProcessor):
    """
    Restricts generation at every step to a per-batch-item allowed token id
    set. `allowed_token_ids_per_item` is indexed by the *source* example
    index (i.e. batch element before beam expansion); `num_beams` is used to
    map a generation batch_id (which enumerates beams) back to its source
    example.
    """

    def __init__(self, allowed_token_ids_per_item, num_beams, vocab_size):
        self.allowed_token_ids_per_item = allowed_token_ids_per_item
        self.num_beams = max(1, num_beams)
        self.vocab_size = vocab_size
        # pre-compute boolean masks lazily per item to avoid rebuilding every step
        self._masks = {}

    def _mask_for(self, item_idx, vocab_size):
        key = (item_idx, vocab_size)
        if key not in self._masks:
            import torch
            mask = torch.full((vocab_size,), float('-inf'))
            allowed = self.allowed_token_ids_per_item[item_idx]
            valid = [i for i in allowed if 0 <= i < vocab_size]
            mask[valid] = 0.0
            self._masks[key] = mask
        return self._masks[key]

    def __call__(self, input_ids, scores):
        vocab_size = scores.shape[-1]
        for batch_id in range(scores.shape[0]):
            item_idx = batch_id // self.num_beams
            mask = self._mask_for(item_idx, vocab_size).to(scores.device)
            scores[batch_id] = scores[batch_id] + mask
        return scores


def build_constrained_logits_processor(tokenizer, task, source_texts, categories, sentiments, num_beams):
    """
    Builds a LogitsProcessorList that restricts decoding, for each source
    example in `source_texts`, to: that example's own input tokens, the
    closed category/sentiment vocabulary, the task's template words, and
    the tokenizer's special tokens.

    Returns None if constrained decoding cannot/should not be applied.
    """
    if not source_texts:
        return None

    template_words = TASK_TEMPLATE_WORDS.get(task, TASK_TEMPLATE_WORDS['gen_scl_nat'])
    fixed_ids = set(_token_ids_for_words(tokenizer, template_words))
    fixed_ids.update(_token_ids_for_words(tokenizer, list(categories)))
    fixed_ids.update(_token_ids_for_words(tokenizer, list(sentiments)))
    fixed_ids.update(_token_ids_for_words(tokenizer, list(string.punctuation)))

    special_ids = {tid for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id,
                                    tokenizer.unk_token_id] if tid is not None}
    fixed_ids.update(special_ids)

    allowed_per_item = []
    for text in source_texts:
        source_ids = set(tokenizer.encode(text, add_special_tokens=False))
        allowed_per_item.append(source_ids | fixed_ids)

    # len(tokenizer) matches the model's resized embedding/logits dimension
    # (the main script calls tfm_model.resize_token_embeddings(len(tokenizer)))
    vocab_size = len(tokenizer)
    processor = ConstrainedVocabProcessor(allowed_per_item, num_beams=num_beams, vocab_size=vocab_size)
    return LogitsProcessorList([processor])
