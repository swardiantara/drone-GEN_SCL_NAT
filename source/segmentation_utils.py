# -*- coding: utf-8 -*-
"""
Optional sentence segmentation at inference time, using ADFLER (Automated
Drone Flight Log Event Recognizer):
  paper: "Automated event recognition in drone flight logs: a noise-robust
          approach" (Silalahi, Ahmad and Studiawan)
  code:  https://github.com/swardiantara/ADFLER
  model: https://huggingface.co/swardiantara/ADFLER-bert-base-cased

ADFLER is a BIOES token-classification (NER) model that performs sentence
boundary detection and Event/NonEvent classification in one pass: each token
is tagged B/I/E/S-Event, B/I/E/S-NonEvent, or O. This is inspired by ATOSS
(https://github.com/ryang1119/ATOSS): segmenting a flight-log message into its
constituent sentences before ASQP inference can simplify the input the
generative model has to reason over. ADFLER additionally lets us drop NonEvent
sentences before ASQP inference: per the paper's annotation criteria (Section
4.2), a NonEvent sentence is "informational, advisory or state-descriptive but
not indicative of a specific event occurring at that moment" and so cannot
contain an aspect-category-sentiment-opinion quadruple by definition. Dropping
those keeps the number of extracted quads aligned with the actual quads in the
message-level input, rather than risking spurious quads generated from
non-informative text.

This only applies to *inference*: segmentation changes what the model sees as
its input at generation time, but training still uses the original,
unsegmented sentences/labels as authored in the dataset files.

Requires the `simpletransformers` package (`pip install simpletransformers`).
"""

DEFAULT_MODEL_DIR = 'swardiantara/ADFLER-bert-base-cased'


def extract_boundaries_with_types(tags):
    """
    Extract (start_idx, end_idx, entity_type) spans from a BIOES tag sequence,
    where entity_type is e.g. "Event" or "NonEvent". Adapted from ADFLER's own
    src/eval_utils.py (the reference implementation the paper's own
    evaluation numbers were produced with), which is robust to malformed tag
    sequences -- a "B-" without a matching "E-", a type change mid-span, etc.
    -- and guarantees every non-"O" token ends up in exactly one span. This
    version additionally closes an open span on an unrecognized tag format
    (treating it like "O"), which the original does not handle.
    """
    boundaries = []
    start_idx = None
    current_entity_type = None

    for idx, tag in enumerate(tags):
        if tag == "O":  # Separator tag; close the current boundary if one is open
            if start_idx is not None:
                boundaries.append((start_idx, idx - 1, current_entity_type))
                start_idx = None
                current_entity_type = None
            continue

        tag_parts = tag.split("-")
        if len(tag_parts) == 2:
            prefix, entity_type = tag_parts
        else:
            # unrecognized tag format; treat like "O"
            if start_idx is not None:
                boundaries.append((start_idx, idx - 1, current_entity_type))
                start_idx = None
                current_entity_type = None
            continue

        if start_idx is None:
            # Begin a new boundary
            start_idx = idx
            current_entity_type = entity_type
        elif entity_type != current_entity_type:
            # Different entity type encountered, close current boundary and start new
            boundaries.append((start_idx, idx - 1, current_entity_type))
            start_idx = idx
            current_entity_type = entity_type
        elif prefix in ("B", "S"):
            # B or S tag appears; start new boundary
            if start_idx is not None:
                boundaries.append((start_idx, idx - 1, current_entity_type))
            start_idx = idx
            current_entity_type = entity_type

    # Close any remaining boundary
    if start_idx is not None:
        boundaries.append((start_idx, len(tags) - 1, current_entity_type))

    return boundaries


class SentenceSegmenter:
    """
    Wraps an ADFLER-style simpletransformers NERModel to segment a raw drone
    log message into its constituent sentences and classify each as Event or
    NonEvent, keeping only the Event sentences for downstream ASQP inference.
    """

    EVENT_TYPE = 'Event'
    NONEVENT_TYPE = 'NonEvent'

    def __init__(self, model_dir, use_cuda=False, model_type='bert'):
        try:
            from simpletransformers.ner import NERModel
        except ImportError as exc:
            raise ImportError(
                "Sentence segmentation requires the 'simpletransformers' package. "
                "Install it with `pip install simpletransformers` to use --use_segmentation."
            ) from exc

        self.model = NERModel(model_type, model_dir, use_cuda=use_cuda, args={'silent': True})

    def _predict_tags(self, messages):
        # simpletransformers' NERModel.predict splits each message on
        # whitespace by default and returns, per message, a list of
        # single-key {word: tag} dicts, one per whitespace token -- matching
        # how ADFLER's own tools/adfler.py calls prediction.
        predictions, _ = self.model.predict(messages)
        words_per_message, tags_per_message = [], []
        for sentence in predictions:
            words = [list(token_dict.keys())[0] for token_dict in sentence]
            tags = [list(token_dict.values())[0] for token_dict in sentence]
            words_per_message.append(words)
            tags_per_message.append(tags)
        return words_per_message, tags_per_message

    def segment(self, messages):
        """
        messages: list[str], one raw (untokenized-joined) message per example.
        Returns list[list[str]]: per-message list of Event sub-sentences only
        -- NonEvent sentences are dropped, since they cannot contain a quad.
        Falls back to the original, unsegmented message as a single segment
        if no Event sentence is found for that message (e.g. the whole
        message is misclassified as NonEvent), so a classifier mistake
        doesn't silently drop real quads from evaluation.
        """
        words_per_message, tags_per_message = self._predict_tags(messages)

        results = []
        for message, words, tags in zip(messages, words_per_message, tags_per_message):
            boundaries = extract_boundaries_with_types(tags)
            event_sentences = [
                ' '.join(words[start:end + 1])
                for start, end, entity_type in boundaries
                if entity_type == self.EVENT_TYPE
            ]
            if not event_sentences:
                event_sentences = [message]
            results.append(event_sentences)
        return results
