# analysis/

Scripts to recap, aggregate, and statistically test the drone ASQP grid
search results produced by `configs/run_drone_paraphrase.sh` /
`configs/run_drone_gen_scl_nat.sh` (via `source/gen_scl_nat_main.py`).

Run all three from the repository root (they read directly from
`train_outputs/` each time, independently of one another):

```
python analysis/recap.py
python analysis/aggregate.py
python analysis/significance.py
```

## Scan scope

All three scripts (via `common.py`) only scan the grid-search output layout:
```
train_outputs/<dataset>/<scenario>/<task>/<absa_task>/cont-{on,off}/cd-{on,off}/seg-{on,off}/<seed>/
```
(`cont-{on,off}` reflects whether `--cont_loss` is nonzero, added so the same
`--task`/template can be compared both with and without the SCL contrastive
loss; older runs from before this existed only have the `cd-*/seg-*` levels
-- `common.py`'s scan matches on the last two path components, so both
layouts are found. `contrastive` itself is always read from `args.json`'s
`cont_loss` value directly, not parsed from the path, so it's correct either
way.) Older/unrelated result folders elsewhere under `train_outputs/` are
ignored.

A run is only included if its `args.json` has a `best_epoch` field, i.e. it
was evaluated using the best-checkpoint-by-validation-F1 selection added to
`gen_scl_nat_main.py`. Runs from before that existed (no `best_epoch`) were
evaluated on whatever the last training epoch left in memory and are not
comparable to the rest of the grid, so they're silently excluded — each
skipped run is printed to stderr so it isn't lost track of silently.

## 1. `recap.py` → `analysis/recap/*.xlsx`

One row per completed run, one file per metric:

| File | Contents |
|---|---|
| `set-PRF.xlsx` | set-based precision/recall/F1, micro/macro/weighted |
| `multiset-PRF.xlsx` | multiset (bag) precision/recall/F1, micro/macro/weighted |
| `generative.xlsx` | BLEU (+ n-gram precisions, brevity penalty, length ratio, lengths), METEOR, ROUGE-1/2/L/Lsum, WER |
| `ac-PRF.xlsx` / `at-PRF.xlsx` / `ot-PRF.xlsx` / `sp-PRF.xlsx` | element-level precision/recall/F1 (micro/macro/weighted) for aspect category / aspect term / opinion term / sentiment polarity, content-matched (multiset) per element -- see `compute_element_scores` in `source/eval_utils.py` |

Every row also carries the config columns: `dataset`, `base_model`,
`template` (`paraphrase` for `--task asqp`, `gen-scl-nat` for
`--task gen_scl_nat`), `contrastive` (`yes` if `--cont_loss > 0`),
`constrained_decoding`, `segmentation`, `seed`.

## 2. `aggregate.py` → `analysis/aggregate/*.xlsx`

Same 7 files, but grouped by every config column except `seed`: for each
metric column `m`, adds `m_mean` and `m_std` (sample std, `ddof=1`, NaN if
only one seed), plus an `n_seeds` column so a configuration missing seeds
(e.g. the excluded run above) is visible rather than silently averaged over
less data.

## 3. `significance.py` → `analysis/statistics/wilcoxon_<metric>.xlsx`

Wilcoxon signed-rank test of every other (template, contrastive,
constrained_decoding, segmentation) scenario against one configurable
baseline scenario, paired by seed, computed separately per (dataset,
base_model) group.

Defaults: metric = multiset micro F1, baseline = paraphrase + contrastive
(the only paraphrase variant run so far), constrained_decoding=no,
segmentation=no. Override via flags, e.g. once paraphrase-without-contrastive
results exist:
```
python analysis/significance.py --baseline_contrastive no
python analysis/significance.py --metric set_micro_f1
```
See `python analysis/significance.py --help` for the full metric list and
baseline flags. Comparisons with fewer than 2 shared seeds, or where every
paired difference is exactly zero, are recorded with `p_value`/`wilcoxon_statistic`
set appropriately and a `note` explaining why (rather than silently omitted).

## 4. `dataset_statistics.py` → `analysis/dataset_statistics.xlsx`

Recaps the raw datasets under `data/` (independent of any run -- reads
`data/<dataset>/{train,dev,test}.txt` directly, no torch dependency), one
row per `(dataset, split)`. Auto-discovers every `data/` subdirectory with
all three split files, or pass `--datasets` to restrict it. Three sheets:

- `overview`: `n_messages`, `n_quads`, `avg_quads_per_message`,
  `n_implicit_aspect`/`pct_implicit_aspect`, `n_implicit_opinion`/`pct_implicit_opinion`
  (aspect/opinion term literally `'NULL'`), `n_full_implicit`/`pct_full_implicit`
  (both implicit in the same quad), `n_unique_sentiment_labels`, `n_unique_categories`.
- `sentiment_distribution`: count + `pct_of_quads` per sentiment label, per `(dataset, split)`.
- `aspect_category_distribution`: count + `pct_of_quads` per category, per `(dataset, split)`.

```
python analysis/dataset_statistics.py
python analysis/dataset_statistics.py --datasets acos_drone_binary acos_drone_multi
```

## 5. `diagnose_segmentation.py` → `analysis/segmentation/<dataset>_<split>_diagnostics.json`

Diagnostic tool for `--use_segmentation` (`source/segmentation_utils.py`):
runs the ADFLER segmentation model on a dataset split's raw messages and
dumps the *token-level* BIOES predictions -- not just the final Event-only
segment list `SentenceSegmenter.segment()` returns during evaluation, but
every token's raw tag, the boundaries `extract_boundaries_with_types()`
derives from them (with Event/NonEvent type), and whether the "zero Event
segments found" whole-message fallback was triggered -- so you can see
exactly what the model is doing per message. Requires `simpletransformers`
(same as `--use_segmentation` itself).

```
python analysis/diagnose_segmentation.py
python analysis/diagnose_segmentation.py --dataset acos_drone_binary --split test
python analysis/diagnose_segmentation.py --segmentation_use_cuda
```

The printed (and saved) `summary` block is the first thing to check:
`tag_distribution` (does the model ever predict `*-NonEvent` tags at all on
this data?), `n_messages_with_zero_boundaries`, and
`n_messages_using_whole_message_fallback` vs. `n_messages` (if this is close
to 100%, the model isn't finding Event spans and everything is falling back
to the unsegmented message -- which is consistent with what the grid search
results currently show: segmentation on vs. off producing identical
predictions across the whole grid).

## Requirements

`pandas`, `numpy`, `scipy`, `openpyxl` (for `.xlsx` output); `diagnose_segmentation.py`
additionally needs `simpletransformers` (and its `torch`/`transformers` dependencies).
