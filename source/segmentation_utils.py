# -*- coding: utf-8 -*-
"""
Optional sentence segmentation at inference time, using LogNexus
(https://pypi.org/project/LogNexs/), inspired by ATOSS
(https://github.com/ryang1119/ATOSS): segmenting a flight-log message into
its constituent sentences before ASQP inference, then merging the per-sentence
predictions back into a single per-message prediction (keeping duplicates),
can simplify the input the generative model has to reason over.

This only applies to *inference*: segmentation changes what the model sees as
its input at generation time, but training still uses the original,
unsegmented sentences/labels as authored in the dataset files.

LogNexus itself requires a separately trained token-classification
(simpletransformers NER-style) segmentation model directory; this module is a
thin, optional wrapper around it so the rest of the pipeline does not need to
depend on `lognexus`/`simpletransformers` unless segmentation is requested.
"""


class SentenceSegmenter:
    def __init__(self, model_dir, use_cuda=False):
        try:
            from lognexus.model import LogNexusModel
        except ImportError as exc:
            raise ImportError(
                "Sentence segmentation requires the 'lognexus' package "
                "(and its 'simpletransformers' runtime dependency). "
                "Install it with `pip install LogNexs` to use --use_segmentation."
            ) from exc

        self.model = LogNexusModel(model_dir, use_cuda=use_cuda)

    def segment(self, messages):
        """
        messages: list[str], one raw (untokenized-joined) message per example.
        Returns list[list[str]]: per-message list of segmented sub-sentences.
        Falls back to the original message as a single segment if the
        segmentation model returns nothing for it.
        """
        segmented = self.model.predict_sentences(messages)
        results = []
        for message, sentences in zip(messages, segmented):
            sentences = [s.strip() for s in sentences if s and s.strip()]
            if not sentences:
                sentences = [message]
            results.append(sentences)
        return results
