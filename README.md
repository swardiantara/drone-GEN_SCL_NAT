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
flags `--use_segmentation --segmentation_model_dir <path>`) — using
[LogNexus](https://pypi.org/project/LogNexs/) (`pip install LogNexs`),
inspired by [ATOSS](https://github.com/ryang1119/ATOSS): each input message is
segmented into sub-sentences, inference is run independently per sub-sentence,
and predicted quadruples are merged back per original message (duplicates
kept, not deduplicated), before scoring against the original gold labels with
both set- and multiset-based metrics (dumped to `results-<dataset>-segmented.json`).
This applies to inference only — training always uses the original,
unsegmented sentences/labels. LogNexus requires a separately trained
segmentation model directory (a `simpletransformers` NER-style checkpoint);
point `--segmentation_model_dir` at it.

**5. Reproducible per-method drone scripts** — `configs/run_drone_paraphrase.sh`
and `configs/run_drone_gen_scl_nat.sh` each train + evaluate one method on a
drone dataset, with `DATASET` / `USE_CONSTRAINED_DECODING` / `USE_SEGMENTATION`
/ `SEGMENTATION_MODEL_DIR` environment-variable toggles for the ablations
above, e.g.:
```
bash configs/run_drone_paraphrase.sh
USE_CONSTRAINED_DECODING=true bash configs/run_drone_paraphrase.sh
USE_SEGMENTATION=true SEGMENTATION_MODEL_DIR=models/lognexus_drone bash configs/run_drone_gen_scl_nat.sh
DATASET=acos_drone_binary bash configs/run_drone_gen_scl_nat.sh
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
