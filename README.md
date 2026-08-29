Code + Models for EMNLP 2022 Findings paper "Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure"
Paper Link: https://arxiv.org/abs/2211.07743

![image](https://user-images.githubusercontent.com/22947125/224211557-5f9b37ac-10ce-4e1b-8c53-a574233d94d1.png)


Pre-trained Model Usage:

The following trained models are available for download on Google Drive (highest-performing model amongst the 5 random seeds):
```
GEN_SCL_NAT-RESTAURANT
GEN_SCL_NAT-LAPTOP
GEN_SCL_NAT-LAPTOP-L1
```

Drive link: https://drive.google.com/drive/folders/1g30oS8hpqn6tAGNyLbOwEoLLmhHOy94o?usp=share_link


Module Requirements:

You can recreate the full Conda environment used by running the following (may require some tweaking of the environment name/path to run on your machine):
```
conda env create -f environment.yml
conda activate gen_scl_nat_env
```

Otherwise, key dependencies used are listed here:

```
Python >= 3.9+
torch >= 1.10
pytorch-lightning >= 1.8.6
sentencepiece >= 0.1.97
transformers >= 4.19.0
```

Module Usage:
1. Initialize + activate conda environment
2. Download and untar trained models to `models/`
3. Run `main_gen_scl_nat.py` for model training/inference. 
	`configs/` contains example scripts for running evaluation on each model from the paper

## Drone forensics extensions

This fork extends the original GEN-SCL-NAT / PARAPHRASE (ABSA-QUAD) codebase for
descriptive-diagnostic quadruple extraction from drone flight-log messages, for
forensic investigation support. Added on top of the pipeline (`source/gen_scl_nat_main.py`,
covering both `--task asqp` and `--task gen_scl_nat`):

**1. Duplicate quadruple analysis** (`source/analyze_duplicates.py`) — quantifies
how often a single example's gold label list contains the exact same quadruple
more than once, per split. Run from the repo root:
```
python source/analyze_duplicates.py --dataset acos_drone_multi --show_examples
```
This is what justifies extension (2) below: on the drone datasets, ~1% of
training examples contain a duplicated gold quadruple, which set-based
evaluation silently collapses.

**2. Set- and multiset-based evaluation** (`source/eval_utils.py`) — every
`--do_train`/`--do_direct_eval`/`--do_inference` run now additionally reports,
under `performance_metrics.set_multiset_scores` in the dumped `results-*.json`
(alongside the existing quad/element-level and generative metrics, untouched):
- **set-based**: duplicates within an example collapsed to one occurrence, and
- **multiset (bag) based**: duplicate quadruples matched with multiplicity
  (via `collections.Counter` intersection), so a repeated gold quadruple must
  be predicted the matching number of times to be fully credited.

Each is reported with **micro** (global TP/gold/pred aggregation), **macro**
(unweighted mean of per-example P/R/F1), and **weighted** (per-example P/R/F1
averaged, weighted by each example's gold-quadruple count) averaging.

**3. Constrained decoding** (`source/constrained_decoding.py`, flag
`--constrained_decoding`) — restricts generation, at every step, to the union
of: the current example's own input tokens (copy vocabulary), the closed
category/sentiment surface-form vocabulary observed in the training targets
(plus the domain's category list from `category_mappings.json` when
available), the task's fixed template tokens (from `category_mappings.json`'s
`special_tokens`, e.g. `is`/`because`/`[SSEP]` for PARAPHRASE,
`|`/`THE`/`IS`/`[SSEP]` for GEN-SCL-NAT), and special tokens. This replaces an
earlier, disabled attempt (`utils.ToggleableConstrainedLogitsProcessor`) that
restricted generation to category/template tokens only, without a source-copy
allowance — which blocks aspect/opinion span tokens from ever being generated.
This is vocabulary-restricted constrained decoding (not a full output-grammar
decoder), intended as an ablation switch.

**4. Optional inference-time sentence segmentation** (`source/segmentation_utils.py`,
flags `--use_segmentation --segmentation_model_dir <path_or_hub_id>
--segmentation_model_type <simpletransformers model_type>`) — using
[ADFLER](https://github.com/swardiantara/ADFLER) (`pip install simpletransformers`;
model card: [swardiantara/ADFLER-bert-base-cased](https://huggingface.co/swardiantara/ADFLER-bert-base-cased)),
inspired by [ATOSS](https://github.com/ryang1119/ATOSS): each input message is
segmented into sentences via ADFLER's BIOES token classification, which
simultaneously classifies each sentence as **Event** or **NonEvent**.
**NonEvent sentences are dropped** before ASQP inference — per the paper's
annotation criteria they are informational/advisory/state-descriptive and
cannot contain a quadruple, so this keeps the number of extracted quads
aligned with the actual quads in the message. If a message ends up with zero
Event sentences (e.g. misclassified entirely as NonEvent), the pipeline falls
back to running inference on the original, unsegmented message so a
classifier mistake doesn't silently drop real quads. Inference is then run
independently per surviving Event sentence, and predicted quadruples are
merged back per original message (duplicates kept, not deduplicated), before
scoring against the original gold labels with both set- and multiset-based
metrics (dumped to `results-<dataset>-segmented.json`). This applies to
inference only — training always uses the original, unsegmented
sentences/labels. `--segmentation_model_dir` defaults to the published
`swardiantara/ADFLER-bert-base-cased` checkpoint (auto-downloaded via
`transformers`), and can be pointed at any other ADFLER-style
`simpletransformers` NER checkpoint (local path or Hub id) instead. For error
analysis, each example's `segment_predictions` field in
`results-<dataset>-segmented.json` records, per surviving Event sentence, its
input text, raw decoded output, and the quadruples parsed from it — i.e. the
prediction *before* it gets merged into the message-level `output_pred`.

**5. Output folder layout & resumable grid search** — since constrained
decoding and segmentation are decode-time-only ablations but the script
trains and evaluates in one shot, each ablation combination gets its own
output folder: `<output_folder>/<dataset>/<scenario>/<task>/<absa_task>/<seed>/cd-{on,off}/seg-{on,off}/`.
`gen_scl_nat_main.py` prints this path (`Output directory for this run: ...`)
as soon as it's determined, so you always know where a run's artifacts are
going. Before training, it also checks whether that folder's `results-*.json`
already exists (the right filename for whether `--use_segmentation` is set)
and, if so, prints `[RESUME] Skipping ...` and exits immediately instead of
re-running — this is what makes the grid search below resumable after a
crash, preemption, or Ctrl-C.

**5b. Best-checkpoint selection** — training now tracks the checkpoint with
the highest validation-set F1 (multiset/bag micro F1, computed via real
beam-search generation on the dev set at the end of every epoch — using the
same `--num_beams`, and the same `--constrained_decoding` setting if enabled,
as the final test evaluation — not just the teacher-forced validation loss),
via a `pl.callbacks.ModelCheckpoint(monitor='val_f1', mode='max')`. After
`trainer.fit()`, that best checkpoint's weights are reloaded before
evaluation, instead of evaluating whatever the last epoch left in memory
(which is what happened before — the previous checkpointing code path
existed but was never actually wired into the `Trainer`, so it was a no-op
regardless of `--early_stopping`). The selected epoch and its validation F1
are recorded as `best_epoch` / `best_val_f1` / `best_checkpoint_path` in
`args.json`. `--early_stopping` remains a separate, optional flag (whether to
stop training early on validation loss) and is unaffected by this. Note this
adds real per-epoch cost (a full beam-search decode over the dev set every
epoch); the raw PyTorch Lightning `.ckpt` used to reload the best weights is
deleted afterward unless `--save_model` is also passed, to avoid leaving a
large checkpoint file behind in every grid search run's folder.

**6. Reproducible per-method grid search scripts** — `configs/run_drone_paraphrase.sh`
and `configs/run_drone_gen_scl_nat.sh` each run the full ablation grid (baseline
/ +constrained-decoding / +segmentation / +both) across 5 fixed seeds — 20
train+eval runs per script — on a drone dataset, printing a pass/skip/fail
summary at the end (and exiting non-zero if anything failed, listing which
combinations to retry). `DATASET` / `SEGMENTATION_MODEL_DIR` /
`SEGMENTATION_MODEL_TYPE` are configurable via environment variables; re-running
either script skips every combination it already completed:
```
bash configs/run_drone_paraphrase.sh
bash configs/run_drone_gen_scl_nat.sh
DATASET=acos_drone_binary bash configs/run_drone_gen_scl_nat.sh
SEGMENTATION_MODEL_DIR=path/to/local/adfler/checkpoint bash configs/run_drone_gen_scl_nat.sh
```

Please cite our paper as such:
```
@InProceedings{peper22generativeacos,
  author = 	"Peper, Joseph J.
			and Wang, Lu",
  title = 	"Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure",
  booktitle = 	"Conference on Empirical Methods in Natural Language Processing",
  year = 	"2022"
}
```
