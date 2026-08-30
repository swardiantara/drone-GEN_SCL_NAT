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
train_outputs/<dataset>/<scenario>/<task>/<absa_task>/<seed>/cd-{on,off}/seg-{on,off}/
```
Older/unrelated result folders elsewhere under `train_outputs/` are ignored.

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
| `ac-PRF.xlsx` / `at-PRF.xlsx` / `ot-PRF.xlsx` / `sp-PRF.xlsx` | element-level precision/recall/F1 for aspect category / aspect term / opinion term / sentiment polarity |

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

## Requirements

`pandas`, `numpy`, `scipy`, `openpyxl` (for `.xlsx` output).
