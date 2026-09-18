# -*- coding: utf-8 -*-
"""
Implementation of Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure, an EMNLP 2022 paper.
If using this codebase / models in your work, please cite our work as follows:

@InProceedings{peper22generativeacos,
  author = 	"Peper, Joseph J.
			and Wang, Lu",
  title = 	"Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure",
  booktitle = 	"Conference on Empirical Methods in Natural Language Processing",
  year = 	"2022"
}

Code is largely derived from the codebase used for the EMNLP 2021 paper "Aspect Sentiment Quad Prediction as Paraphrase Generation"
https://github.com/IsakZhang/ABSA-QUAD
"""
import argparse
import os
import sys
import logging
import time
import pickle
from tqdm import tqdm
import json
import numpy as np
import copy

from torch import nn
from torch.nn.functional import normalize, mse_loss
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from losses import SupConLoss

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, EncoderDecoderModel
from transformers import get_linear_schedule_with_warmup

from data_utils import GenSCLNatDataset, get_transformed_io
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores, compute_gen_metrics, extract_spans_para
from utils import load_mappings
from constrained_decoding import build_label_vocab, build_constrained_logits_processor
from segmentation_utils import PySBDSegmenter

mappings = load_mappings()
logger = logging.getLogger(__name__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [`asqp`, `gen_scl_nat`]")
    parser.add_argument("--absa_task", default='quad', type=str, required=False,
                        help="The name of the ABSA task, selected from: [`quad`, `tasd`, `aste`, `acsp`, `atsp`]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [`rest15`, `rest16`]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    # other parameters
    parser.add_argument("--accelerator", default='gpu', type=str,
                        help="Device for accelerator: [`cpu`, `gpu`]")
    parser.add_argument('--scenario',  default='t5', required=True,
                        help="Model scenario to fine-tune for paraphrasing task. Default: `t5`")
    # choices=['t5', 'flan-t5', 'flan-t5-large', 'bert2gpt2', 'bert2bert', 'roberta2roberta', 'roberta2gpt2'],
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--num_beams", type=int, required=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.0)
    parser.add_argument("--cont_temp", type=float, default=0.1)
    parser.add_argument("--quad_count_loss", type=float, default=0.0,
                         help="Scaling factor for the quad-count regression auxiliary loss (MSE between "
                              "the true number of quadruples in the example and a scalar predicted from "
                              "the pooled encoder representation). 0.0 (default) disables it, matching "
                              "--cont_loss's convention -- see T5FineTuner._step / QuadCountRegressor.")
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--constrained_decoding', action='store_true',
                        help="Restrict generation to source-copy tokens + closed category/sentiment "
                             "vocabulary (from training targets + category_mappings.json) + template tokens.")
    parser.add_argument('--use_segmentation', action='store_true',
                        help="Segment each input message into sentences (via PySBD) before inference, run "
                             "inference per sentence, then merge predictions per message (keeping "
                             "duplicates). Inference-only; has no effect on training. See "
                             "source/segmentation_utils.py's PySBDSegmenter.")
    parser.add_argument('--also_eval_segmented', action='store_true',
                        help="Segmentation only changes inference, not training (see --use_segmentation), so "
                             "training a seg-off and a seg-on scenario that are otherwise identical is "
                             "redundant -- this trains ONCE (as a seg-off run) and then, on that same trained "
                             "checkpoint, runs a SECOND evaluation pass with segmentation on, writing its "
                             "results into the sibling seg-on/ output folder exactly as if it had been trained "
                             "separately (see init_args' segmented_output_dir). Requires --use_segmentation to "
                             "NOT be set (this flag's whole point is to produce the seg-on results as a "
                             "byproduct of the seg-off run).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Bypass the resume check below and rerun even if this scenario's "
                             "results-*.json already exists (e.g. to retrain with --save_model "
                             "after an earlier run of the same scenario didn't save weights).")

    args = parser.parse_args()

    # create output folder if needed
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    params = [['beams', str(args.num_beams)],
              ['wd', str(args.weight_decay)],
              ['max_epochs', str(args.num_train_epochs)],
              ['es', str(args.early_stopping)],
              ['acc', str(args.gradient_accumulation_steps)],
              ['lr', str(args.learning_rate)],
              ['cont_loss', str(args.cont_loss)],
              ['cont_temp', str(args.cont_temp)],
              ['quad_count_loss', str(args.quad_count_loss)],
              ['trunc', str(args.truncate)], # whether to truncate the category labels
              ['seed', str(args.seed)]]

    # contrastive loss (train-time), and constrained decoding / segmentation
    # (decode-time-only) are all ablations, but this script always trains +
    # evaluates in one shot, so each combination needs its own output folder
    # to be resumable / not clobber another combination's results (see the
    # grid search loops in configs/run_drone_paraphrase.sh /
    # configs/run_drone_gen_scl_nat.sh). cont-{on,off} reflects whether
    # --cont_loss is nonzero, i.e. whether the SCL auxiliary loss actually
    # contributes to training (see T5FineTuner._step) -- this lets the same
    # --task (paraphrase template or gen-scl-nat template) be compared both
    # with and without contrastive learning.
    # --quad_count_loss (whether the quad-count regression auxiliary loss --
    # see T5FineTuner._step / QuadCountRegressor -- contributes to training)
    # is, unlike the other ablations above, encoded ASYMMETRICALLY: an extra
    # 'qc-on' folder segment is appended only when it's actually on; when
    # it's off (the default, 0.0), NO segment is added at all. This is
    # deliberate, not an oversight: --quad_count_loss was added after every
    # existing scenario had already been trained+evaluated under
    # cont-*/cd-*/seg-*/<seed>/ (no qc segment). Encoding qc-off as its own
    # segment (e.g. 'qc-off') would have put those *same* scenarios at a
    # *different* path than where their completed results already sit, so
    # the resume check below would never find them and would silently
    # retrain+re-evaluate all of them from scratch. Appending a segment only
    # for the new qc-on scenarios keeps every previously-completed run's path
    # (and its resumability) exactly as it was. See common.py's
    # find_run_dirs() for the matching scan patterns, which read qc-on from
    # this segment when present and qc-off from --quad_count_loss in
    # args.json directly when it's absent.
    if args.also_eval_segmented and args.use_segmentation:
        raise ValueError("--also_eval_segmented trains a seg-off scenario and derives its seg-on companion "
                          "from the same checkpoint -- pass it without --use_segmentation.")

    ablation_parts = [
        'cont-{}'.format('on' if float(args.cont_loss) > 0.0 else 'off'),
        'cd-{}'.format('on' if args.constrained_decoding else 'off'),
        'seg-{}'.format('on' if args.use_segmentation else 'off'),
    ]
    if float(args.quad_count_loss) > 0.0:
        ablation_parts.append('qc-on')
    ablation_tag = os.path.join(*ablation_parts)

    # TODO CLEANUP TRAINING OUTPUT FOLDER
    # the model path is the prefix
    if args.do_inference and not args.do_train:
        output_fold = os.path.join(args.model_prefix, ablation_tag)
        print(output_fold)
    else:
        # dump params as part of folder_path
        # params = "I".join([elt for elts in params for elt in elts])
        # output_fold = "I".join([args.dataset, args.task,args.model_name_or_path, params, args.model_prefix])
        # output_fold = "_".join([args.dataset, args.task, args.model_prefix, args.model_name_or_path])
        # seed is last so all seeds for one scenario sit side by side on disk
        output_fold = os.path.join(args.dataset, args.scenario, args.task, args.absa_task, ablation_tag, str(args.seed))

        print(output_fold)
    output_dir = os.path.join(args.output_folder, output_fold)

    # the result file name differs between the segmented and standard eval
    # pipelines (see evaluate() / evaluate_segmented()); check for whichever
    # one this run would produce, so a completed run of either kind is
    # correctly detected and skipped on resume.
    result_filename = (f'results-{args.dataset}-segmented.json' if args.use_segmentation
                        else f'results-{args.dataset}.json')

    # --also_eval_segmented's companion seg-on output sits at the exact same
    # path a standalone `--use_segmentation` run of this scenario would use
    # (same cont/cd/qc/seed, 'seg-off' swapped for 'seg-on') -- computed here
    # regardless of whether the flag is set, since it's cheap and only ever
    # read when it's on.
    segmented_ablation_parts = list(ablation_parts)
    segmented_ablation_parts[2] = 'seg-on'
    segmented_ablation_tag = os.path.join(*segmented_ablation_parts)
    segmented_output_fold = os.path.join(args.dataset, args.scenario, args.task, args.absa_task,
                                          segmented_ablation_tag, str(args.seed))
    args.segmented_output_dir = os.path.join(args.output_folder, segmented_output_fold)
    segmented_result_path = os.path.join(args.segmented_output_dir, f'results-{args.dataset}-segmented.json')

    already_done = os.path.exists(os.path.join(output_dir, result_filename))
    if args.also_eval_segmented:
        already_done = already_done and os.path.exists(segmented_result_path)
    if already_done and not args.overwrite:
        print(f'[RESUME] Skipping {output_dir}: already completed '
              f'(found {result_filename}' +
              (f' and {segmented_result_path}' if args.also_eval_segmented else '') + ')')
        sys.exit(0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if args.also_eval_segmented and not os.path.exists(args.segmented_output_dir):
        os.makedirs(args.segmented_output_dir)
    args.output_dir = output_dir
    print(f"Output directory for this run: {args.output_dir}")

    return args


def get_dataset(tokenizer, type_path, args):
    return GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length, task=args.task, absa_task=args.absa_task, truncate=args.truncate)

"""
Uncomment for tsne logging
tsne_dict = {
             'sentiment_vecs': [],
             'opinion_vecs': [],
             'aspect_vecs': [],
             'sentiment_labels': [],
             'opinion_labels': [],
             'aspect_labels': []
             }
"""

class LinearModel(nn.Module):
    """
    Linear models used for the aspect/opinion/sentiment-specific representations
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, 1024)
        # if model_path == 't5-small':
        #     self.layer_1 = nn.Linear(512, 1024)
        # elif model_path == 't5-base':
        #     self.layer_1 = nn.Linear(768, 1024)
        # else:
        #     self.layer_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        Returns an encoding of input X and a simple dropout-perturbed version of X
        For use in the SupConLoss calculation
        """
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)


class QuadCountRegressor(nn.Module):
    """
    Predicts the number of quadruples in an example (a scalar regression
    target, see get_n_quads_labels in data_utils.py) from the pooled encoder
    representation (T5FineTuner.forward's pooled_encoder_layer -- the same
    masked-sum-pooled, L2-normalized representation the sentiment/aspect/
    opinion SCL heads use; T5 has no BERT-style [CLS] token, so there is no
    literal classification-token hidden state to use instead).
    """
    def __init__(self, d_model=768):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(0.1)
        self.layer_2 = nn.Linear(256, 1)

    def forward(self, pooled_encoder_layer):
        x = self.dropout(torch.relu(self.layer_1(pooled_encoder_layer)))
        return self.layer_2(x).squeeze(-1)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, seq2seq_model, tokenizer, cont_model, op_model, as_model, qc_model):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = seq2seq_model
        self.cont_model = cont_model
        self.op_model = op_model
        self.as_model = as_model
        # self.cat_model = cat_model
        self.qc_model = qc_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
       
        last_state = main_pred.encoder_last_hidden_state

        # sentiment contrastive loss
        cont_pred = self.cont_model(last_state, attention_mask)
        # opinion contrastive loss
        op_pred = self.op_model(last_state, attention_mask)
        # aspect contrastive loss
        as_pred = self.as_model(last_state, attention_mask)
        
        # get final encoder layer representation
        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        # quad-count regression (auxiliary task, scaled by --quad_count_loss)
        qc_pred = self.qc_model(pooled_encoder_layer)

        return main_pred, cont_pred, op_pred, as_pred, pooled_encoder_layer, qc_pred


    def _step(self, batch):
        lm_labels = torch.clone(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, cont_pred, op_pred, as_pred, pooled_encoder_layer, qc_pred = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        # define loss with a temperature `temp`
        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        sentiment_labels = batch['sentiment_labels']
        aspect_labels = batch['aspect_labels']
        opinion_labels = batch['opinion_labels']

        # Calculate the characteristic-specific losses
        cont_summed = cont_pred
        cont_normed = normalize(cont_summed, p=2.0, dim=2)  
        sentiment_contrastive_loss = criterion(cont_normed, sentiment_labels)
        #print('contr_loss:\t', sentiment_contrastive_loss)

        as_summed = as_pred
        as_normed = normalize(as_summed, p=2.0, dim=2)
        aspect_contrastive_loss = criterion(as_normed, aspect_labels)
        #print('as_loss:\t', aspect_contrastive_loss)

        op_summed = op_pred
        op_normed = normalize(op_summed, p=2.0, dim=2)
        opinion_contrastive_loss = criterion(op_normed, opinion_labels)
        #print('op_loss:\t', opinion_contrastive_loss)

        # quad-count regression auxiliary loss: MSE between the predicted
        # and true number of quadruples, scaled by --quad_count_loss (0.0
        # makes its contribution exactly zero, same on/off convention as
        # --cont_loss above)
        n_quads_labels = batch['n_quads_labels']
        quad_count_loss = self.hparams.quad_count_loss * mse_loss(qc_pred, n_quads_labels)


        """
        Uncomment this section to extract the tsne encodings/labels used for Figure 2 in paper

        # Use these for generating the 'w/ SCL' figures
        sentiment_encs = cont_normed.detach().numpy()[:,0].tolist()
        aspect_encs = as_normed.detach().numpy()[:,0].tolist()
        opinion_encs = op_normed.detach().numpy()[:,0].tolist()
        sentiment_labs = sentiment_labels.detach().tolist()
        aspect_labs = aspect_labels.detach().tolist()
        opinion_labs = opinion_labels.detach().tolist()

        # Use these for the version without SCL (no characteristic-specific representations)
        
        sentiment_encs = pooled_encoder_layer.detach().numpy().tolist()
        aspect_encs = pooled_encoder_layer.detach().numpy().tolist()
        opinion_encs = pooled_encoder_layer.detach().numpy().tolist()
        sentiment_labs = sentiment_labels.detach().tolist()
        aspect_labs = aspect_labels.detach().tolist()
        opinion_labs = opinion_labels.detach().tolist()

        tsne_dict['sentiment_vecs'] += sentiment_encs
        tsne_dict['aspect_vecs'] += aspect_encs
        tsne_dict['opinion_vecs'] += opinion_encs
        tsne_dict['sentiment_labels'] += sentiment_labs
        tsne_dict['aspect_labels'] += aspect_labs
        tsne_dict['opinion_labels'] += opinion_labs
        """

        # return original loss plus the characteristic-specific SCL losses
        # and the quad-count regression auxiliary loss
        loss = (outputs[0] + opinion_contrastive_loss + sentiment_contrastive_loss
                + aspect_contrastive_loss + quad_count_loss)
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log('avg_train_loss', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss, pred_outputs = self._step(batch)
        self.log('val_batch_loss', loss)
        preds, targets = self._generate_val_predictions(batch)
        return {"val_batch_loss": loss, "preds": preds, "targets": targets}

    def _get_constrained_decoding_vocab(self):
        """
        Lazily builds and caches the constrained-decoding vocabulary (see
        constrained_decoding.py) once per training run, since it's derived
        from the whole training set and is expensive to recompute every
        validation batch/epoch.
        """
        if not hasattr(self, '_cd_category_vocab'):
            self._cd_category_vocab, self._cd_sentiment_vocab = build_label_vocab(
                self.hparams.task, self.hparams.absa_task, self.hparams.dataset, self.hparams.truncate)
            try:
                self._cd_extra_category_words = get_aspect_category(self.hparams)
            except (KeyError, IndexError):
                self._cd_extra_category_words = None
        return self._cd_category_vocab, self._cd_sentiment_vocab, self._cd_extra_category_words

    def _generate_val_predictions(self, batch):
        """
        Runs actual beam-search generation (not just teacher-forced loss) on
        a validation batch, using the same --num_beams (and, if enabled, the
        same constrained decoding) as the final test evaluation -- so the
        F1 used for best-checkpoint selection matches how the model will
        actually be decoded at test time.
        """
        gen_kwargs = dict(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            max_length=self.hparams.max_seq_length * 2,
            num_beams=self.hparams.num_beams,
        )

        if self.hparams.constrained_decoding:
            category_vocab, sentiment_vocab, extra_category_words = self._get_constrained_decoding_vocab()
            source_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['source_ids']]
            logits_processor = build_constrained_logits_processor(
                self.tokenizer, self.hparams.task, source_texts, category_vocab, sentiment_vocab,
                self.hparams.num_beams, mappings=mappings, extra_category_words=extra_category_words)
            if logits_processor is not None:
                gen_kwargs['logits_processor'] = logits_processor

        outs = self.model.generate(**gen_kwargs)
        preds = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['target_ids']]
        return preds, targets

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_batch_loss"] for x in outputs]).mean()
        print("val_loss:\t", avg_loss )
        self.log('val_loss', avg_loss)

        all_preds = [pred for x in outputs for pred in x["preds"]]
        all_targets = [target for x in outputs for target in x["targets"]]
        scores, _, _ = compute_scores(all_preds, all_targets, self.hparams.task, self.hparams.absa_task, silent=True)
        # multiset (bag) micro F1: the metric used to select the best checkpoint
        # (see the ModelCheckpoint(monitor='val_f1', ...) callback in __main__)
        val_f1 = scores['set_multiset_scores']['multiset']['micro']['f1']
        print("val_f1 (multiset micro):\t", val_f1)
        self.log('val_f1', val_f1)

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        cont_model = self.cont_model
        op_model = self.op_model
        as_model = self.as_model
        qc_model = self.qc_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in qc_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in qc_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.callback_metrics['avg_loss']), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def get_aspect_category(args):
    domain = args.dataset.split('_')[1]
    acs = mappings[f'{domain}_full_mapping']
    categories = [ac[1] for ac in acs] if args.task == 'asqp' else [ac[0] for ac in acs]    # later for gen_scl_nat-based aspect category
    return categories


def evaluate(data_loader, model, device, tokenizer, sents, args, category_vocab=None, sentiment_vocab=None):
    """
    Compute scores given the predictions and gold labels and dump to file
    """
    model.model.to(device)

    model.eval()
    model.model.eval()

    extra_category_words = None
    if args.constrained_decoding:
        try:
            extra_category_words = get_aspect_category(args)
        except (KeyError, IndexError):
            # domain not present in category_mappings.json; fall back to the
            # empirically-derived category vocabulary only
            extra_category_words = None

    outputs, targets = [], []
    for batch in tqdm(data_loader):

        gen_kwargs = dict(input_ids=batch['source_ids'].to(device),
                           attention_mask=batch['source_mask'].to(device),
                           max_length=args.max_seq_length * 2,
                           num_beams=args.num_beams)

        if args.constrained_decoding:
            source_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['source_ids']]
            logits_processor = build_constrained_logits_processor(
                tokenizer, args.task, source_texts, category_vocab, sentiment_vocab, args.num_beams,
                mappings=mappings, extra_category_words=extra_category_words)
            if logits_processor is not None:
                gen_kwargs['logits_processor'] = logits_processor

        outs = model.model.generate(**gen_kwargs)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    scores, all_labels, all_preds = compute_scores(outputs, targets, args.task, args.absa_task, False)
    results = {'labels_correct': all_labels, 'labels_pred': all_preds, 'output_pred': outputs, 'output_correct': targets, 'utterances': sents}
    gen_scores = compute_gen_metrics(outputs, targets, False)
    ex_list = []

    for idx in range(len(all_preds)):
        new_dict = {}
        for key in results:
            new_dict[key] = results[key][idx]
        ex_list.append(new_dict)

    results = {'generative_score': gen_scores, 'performance_metrics': scores, 'examples': ex_list}

    json.dump(results, open(f"{args.output_dir}/results-{args.dataset}.json", 'w'), indent=2, sort_keys=True)
    return scores


def evaluate_segmented(model, device, tokenizer, sents, args, category_vocab=None, sentiment_vocab=None):
    """
    Optional inference-time pipeline: segment each input message into
    sentences (PySBD), run inference independently per sentence, then merge
    the predicted quadruples back per original message (concatenation, so
    duplicate quadruples are preserved), and score against the gold
    quadruples for the *original* (unsegmented) message.
    """
    model.model.to(device)
    model.eval()
    model.model.eval()

    # gold quadruples: derived from the standard, unsegmented target-text
    # formatting/parsing pipeline, so scoring stays comparable to non-segmented runs
    data_path = f'data/{args.dataset}/test.txt'
    _, gold_targets, _ = get_transformed_io(data_path, args.dataset, args.task, args.absa_task, 'test', args.truncate)
    gold_target_texts = [t if isinstance(t, str) else " ".join(t) for t in gold_targets]

    extra_category_words = None
    if args.constrained_decoding:
        try:
            extra_category_words = get_aspect_category(args)
        except (KeyError, IndexError):
            extra_category_words = None

    segmenter = PySBDSegmenter()
    messages = [' '.join(sent) for sent in sents]
    segmented = segmenter.segment(messages)

    per_example_outputs = []
    # per-message list of per-segment prediction details, for error analysis:
    # what exactly each individual Event sentence was fed and what it predicted,
    # before it gets concatenated/merged into the message-level prediction above
    per_example_segment_predictions = []
    for segments in tqdm(segmented):
        merged_texts = []
        segment_predictions = []
        for segment_text in segments:
            tokenized = tokenizer.batch_encode_plus(
                [segment_text], max_length=args.max_seq_length, padding='max_length',
                truncation=True, return_tensors='pt')

            gen_kwargs = dict(input_ids=tokenized['input_ids'].to(device),
                               attention_mask=tokenized['attention_mask'].to(device),
                               max_length=args.max_seq_length * 2,
                               num_beams=args.num_beams)

            if args.constrained_decoding:
                logits_processor = build_constrained_logits_processor(
                    tokenizer, args.task, [segment_text], category_vocab, sentiment_vocab, args.num_beams,
                    mappings=mappings, extra_category_words=extra_category_words)
                if logits_processor is not None:
                    gen_kwargs['logits_processor'] = logits_processor

            outs = model.model.generate(**gen_kwargs)
            dec = tokenizer.decode(outs[0], skip_special_tokens=True)
            merged_texts.append(dec)
            segment_predictions.append({
                'segment_text': segment_text,
                'predicted_output': dec,
                'predicted_quads': extract_spans_para(args.task, args.absa_task, dec, 'pred'),
            })

        # concatenating each segment's (possibly multi-quad) decoded text with
        # [SSEP] preserves duplicate quadruples across segments once parsed
        per_example_outputs.append(' [SSEP] '.join(merged_texts))
        per_example_segment_predictions.append(segment_predictions)

    # reuses the same extraction/scoring path as the non-segmented pipeline
    # (compute_scores), so all_preds/all_labels here are quads parsed from the
    # merged per-message text via extract_spans_para, duplicates preserved
    quad_scores, all_labels, all_preds = compute_scores(per_example_outputs, gold_target_texts, args.task, args.absa_task, False)
    gen_scores = compute_gen_metrics(per_example_outputs, gold_target_texts, False)

    results = {'labels_correct': all_labels, 'labels_pred': all_preds, 'output_pred': per_example_outputs,
               'utterances': sents, 'segments': segmented, 'segment_predictions': per_example_segment_predictions}
    ex_list = []
    for idx in range(len(all_preds)):
        new_dict = {}
        for key in results:
            new_dict[key] = results[key][idx]
        ex_list.append(new_dict)

    results = {'generative_score': gen_scores, 'performance_metrics': quad_scores, 'examples': ex_list}
    json.dump(results, open(f"{args.output_dir}/results-{args.dataset}-segmented.json", 'w'), indent=2, sort_keys=True)
    return quad_scores


def run_evaluation(data_loader, model, device, tokenizer, sents, args):
    """
    Dispatches to the segmented or standard evaluation pipeline based on
    --use_segmentation, wiring up constrained decoding (--constrained_decoding)
    for either path.
    """
    category_vocab, sentiment_vocab = None, None
    if args.constrained_decoding:
        category_vocab, sentiment_vocab = build_label_vocab(args.task, args.absa_task, args.dataset, args.truncate)

    if args.use_segmentation:
        return evaluate_segmented(model, device, tokenizer, sents, args, category_vocab, sentiment_vocab)
    return evaluate(data_loader, model, device, tokenizer, sents, args, category_vocab, sentiment_vocab)


def get_seq2seq_model(args):
    scenarios = args.scenario.split('_')
    # initialize the tokenizer and seq2seq model
    if scenarios[0] in ['drone', 'laptop', 'restaurant']:
        emb_scenario = '_'.join([scenarios[2], scenarios[3], scenarios[4]])
        model_path = os.path.join('embeddings', f"acos_{scenarios[0]}_{'data' if scenarios[0] != 'drone' else scenarios[1]}", emb_scenario)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_path, model_path)
    elif str(args.scenario).startswith('t5'):
        args.model_name_or_path = f'google-t5/{args.scenario}'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif str(args.scenario).startswith('flan'):
        args.model_name_or_path = f'google/{args.scenario}'
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.scenario == 'bert2gpt2':
        args.model_name_or_path = 'bert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "gpt2")
    elif args.scenario == 'bert2bert':
        args.model_name_or_path = 'bert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
        seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained("google-bert/bert-base-cased", "google-bert/-base-cased")
    elif args.scenario == 'roberta2gpt2':
        args.model_name_or_path = 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained("FacebookAI/roberta-base", "gpt2")
    elif args.scenario == 'roberta2roberta':
        args.model_name_or_path = 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        seq2seq_model = EncoderDecoderModel.from_encoder_decoder_pretrained("FacebookAI/roberta-base", "FacebookAI/roberta-base")
    else:
        raise NotImplementedError
    
    return tokenizer, seq2seq_model

    
# check for top-level environment
if __name__ == '__main__':
    # initialization
    args = init_args()
    seed_everything(args.seed, workers=True)
    device = torch.device('cpu' if args.accelerator == 'cpu' else 'cuda')

    tokenizer, seq2seq_model = get_seq2seq_model(args)
    tokenizer.add_tokens(['[SSEP]'])
    # Get example from the train set
    dataset = GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                        data_type='train', max_len=args.max_seq_length, task=args.task, absa_task=args.absa_task, truncate=args.truncate)
    data_sample = dataset[0]

    # sanity check
    # show one sample to check the code and the expected output format are correct
    print(f"Here is an example (from the train set):")
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print(data_sample['source_ids'])
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
    print(data_sample['target_ids'])

    # training process
    if args.do_train:
        print("\n****** Conducting Training ******")

        if str(args.scenario).startswith('t5') or str(args.scenario).startswith('flan'):
            seq2seq_model.resize_token_embeddings(len(tokenizer))
        else:
            # Adjust special token's IDs
            seq2seq_model.encoder.resize_token_embeddings(len(tokenizer))
            seq2seq_model.decoder.resize_token_embeddings(len(tokenizer))
            seq2seq_model.config.decoder_start_token_id = tokenizer.cls_token_id
            seq2seq_model.config.eos_token_id = tokenizer.sep_token_id
            seq2seq_model.config.pad_token_id = tokenizer.pad_token_id
            seq2seq_model.config.vocab_size = seq2seq_model.config.encoder.vocab_size


        # initialize characteristic-specific representation models
        cont_model = LinearModel()
        op_model = LinearModel()
        as_model = LinearModel()
        # cat_model = LinearModel()
        qc_model = QuadCountRegressor()
        model = T5FineTuner(args, seq2seq_model, tokenizer, cont_model, op_model, as_model, qc_model)

        # always track the best checkpoint by validation-set (multiset micro)
        # F1 -- see T5FineTuner.validation_epoch_end -- so evaluation uses the
        # best-performing epoch rather than whatever the last epoch happened
        # to be. --early_stopping is a separate, optional concern (whether to
        # stop training early); it still monitors val_loss as before.
        checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=args.output_dir, filename='best-{epoch}-{val_f1:.4f}',
            monitor='val_f1', mode='max', save_top_k=1
        )
        callback_list = [checkpoint_callback, LoggingCallback()]
        if args.early_stopping:
            callback_list.append(EarlyStopping(monitor="val_loss", mode='min', patience=3))

        # prepare trainer args
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            # accelerator=args.accelerator,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            auto_lr_find=False,
            deterministic=True,
            logger=None,
            #auto_scale_batch_size=True,
            callbacks=callback_list,
        )
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        # reload the best (highest val_f1) checkpoint's weights for evaluation,
        # rather than evaluating with whatever the last epoch left in memory
        best_checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(best_checkpoint['state_dict'])
        args.best_epoch = best_checkpoint.get('epoch')
        args.best_val_f1 = float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None
        args.best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Best checkpoint: epoch {args.best_epoch}, val_f1={args.best_val_f1} ({args.best_checkpoint_path})")

        if args.save_model:
            model.model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        else:
            # the raw PyTorch Lightning checkpoint (full T5 + auxiliary SCL
            # heads) is only needed transiently, to select and reload the
            # best epoch's weights above; without --save_model there's no
            # HF-format checkpoint kept either, so delete it rather than
            # leaving a large .ckpt behind in every grid search run's folder
            if os.path.exists(args.best_checkpoint_path):
                os.remove(args.best_checkpoint_path)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("Finish training and saving the model!")

    # evaluation
    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        test_dataset = GenSCLNatDataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length, task=args.task, absa_task=args.absa_task, truncate=args.truncate)
        test_loader = DataLoader(test_dataset, args.eval_batch_size, num_workers=4)

        # compute the performance scores
        run_evaluation(test_loader, model, device, tokenizer, test_dataset.sentence_strings, args)

        if args.also_eval_segmented:
            # same trained model, second evaluation pass with segmentation on
            # -- writes into the seg-on sibling folder (args.segmented_output_dir,
            # set in init_args) so it's indistinguishable from a standalone
            # seg-on run's output, without retraining
            print("\n****** Conduct Evaluating (segmented companion, same checkpoint) ******")
            segmented_args = copy.deepcopy(args)
            segmented_args.use_segmentation = True
            segmented_args.also_eval_segmented = False
            segmented_args.output_dir = args.segmented_output_dir
            run_evaluation(test_loader, model, device, tokenizer, test_dataset.sentence_strings, segmented_args)
            with open(os.path.join(segmented_args.output_dir, 'args.json'), 'w') as f:
                json.dump(segmented_args.__dict__, f, indent=2)

    if args.do_inference:
        print("\n****** Conduct inference on trained checkpoint ******")

        if not args.do_train:
            # initialize the seq2seq model from previous checkpoint
            if args.scenario == 't5':
                print(f"Loading trained model from {args.model_name_or_path}")
                tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
                seq2seq_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            else:
                raise NotImplementedError

        # representations are only used during loss calculation
        cont_model = LinearModel()
        op_model = LinearModel()
        as_model = LinearModel()
        # cat_model = LinearModel()
        qc_model = QuadCountRegressor()
        model = T5FineTuner(args, seq2seq_model, tokenizer, cont_model, op_model, as_model, qc_model)

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        test_dataset = GenSCLNatDataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length, task=args.task, absa_task=args.absa_task, truncate=args.truncate)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)

        # compute the performance scores
        run_evaluation(test_loader, model, device, tokenizer, test_dataset.sentence_strings, args)
    