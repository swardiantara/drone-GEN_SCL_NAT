#!/bin/bash
# Grid search entry point for the GEN-SCL-NAT (--task gen_scl_nat) method on
# the drone forensics dataset: one invocation runs every scenario -- base
# model x contrastive-loss on/off x constrained-decoding on/off x
# quad-count-regression-loss on/off -- across 10 seeds. Default grid: 2 base
# models x 2 contrastive x 2 CD x 2 quad-count x 10 seeds = 160 train+eval
# runs. Segmentation is NOT a separate combination here: it only changes
# inference (source/segmentation_utils.py's PySBDSegmenter has no effect on
# training), so RUN_SEGMENTATION=true (the default) adds
# --also_eval_segmented to every run instead of doubling the grid -- each
# scenario trains once and gets a second, segmented evaluation pass on that
# same checkpoint, written straight into its seg-on/ sibling folder. Set
# RUN_SEGMENTATION=false to skip that extra eval pass entirely.
#
# Resumable: source/gen_scl_nat_main.py's own output-folder check (keyed by
# dataset/scenario/task/absa_task/cont-{on,off}/cd-{on,off}/seg-{on,off}/
# [qc-on/]seed) skips any combination whose results-*.json already exists,
# so re-running this script after it stopped partway through (crash,
# preemption, Ctrl-C, ...) picks up where it left off instead of redoing
# completed runs. quad_count_loss=0.0 (qc-off) runs land at the SAME path as
# before this axis existed (.../seg-{on,off}/seed/, no qc-* segment at all --
# a 'qc-on' segment is added only when the auxiliary task is actually on),
# so scenarios you already ran before this axis was added are correctly
# skipped/resumed rather than being retrained under a new path. If you've
# deleted outputs/ to start clean (e.g. after a dataset fix), every
# combination runs fresh.
#
# Usage:
#   bash configs/run_drone_gen_scl_nat.sh
#   DATASET=acos_drone_binary bash configs/run_drone_gen_scl_nat.sh
#   BASE_MODELS="t5-base flan-t5-base" bash configs/run_drone_gen_scl_nat.sh
#   CONT_LOSS_OPTIONS="0.0 0.05" bash configs/run_drone_gen_scl_nat.sh
#   QUAD_COUNT_LOSS_OPTIONS="0.0 1.0" bash configs/run_drone_gen_scl_nat.sh
#   RUN_SEGMENTATION=true bash configs/run_drone_gen_scl_nat.sh

set -uo pipefail

DATASET=${DATASET:-acos_drone_binary}
ABSA_TASK=${ABSA_TASK:-quad}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-outputs}
MODEL_PREFIX=${MODEL_PREFIX:-drone_gen_scl_nat}

# base-model scenarios to sweep (space-separated; each becomes --scenario,
# see source/gen_scl_nat_main.py's get_seq2seq_model() for the full list of
# recognized values, e.g. t5-base, flan-t5, flan-t5-large, bert2gpt2, ...)
read -ra BASE_MODELS <<< "${BASE_MODELS:-t5-base flan-t5-base}"

# --cont_loss/--cont_temp: T5FineTuner._step always computes the SCL
# auxiliary loss, scaled by --cont_loss -- 0.0 makes its contribution exactly
# zero (see source/losses.py's SupConLoss), i.e. contrastive learning OFF,
# which isolates the effect of the gen-scl-nat target template itself.
# gen_scl_nat_main.py's output folder already encodes this as cont-{on,off}
# (derived from --cont_loss > 0), so each cont_loss value lands in its own
# folder and is resumable/isolated independently.
read -ra CONT_LOSS_OPTIONS <<< "${CONT_LOSS_OPTIONS:-0.0 0.05}"
CONT_TEMP=${CONT_TEMP:-0.25}

# --quad_count_loss: T5FineTuner._step always computes the quad-count
# regression MSE loss (predicting the number of quadruples in the example
# from the pooled encoder representation), scaled by --quad_count_loss --
# 0.0 makes its contribution exactly zero, i.e. this auxiliary task OFF.
# Unlike cont-{on,off}, gen_scl_nat_main.py's output folder encodes this
# ASYMMETRICALLY: a 'qc-on' segment is appended only when nonzero; 0.0 (off)
# adds no segment at all, landing at the exact same path a pre-quad-count-
# task run would -- see init_args() in gen_scl_nat_main.py for why (keeps
# every already-completed scenario's resume check working).
read -ra QUAD_COUNT_LOSS_OPTIONS <<< "${QUAD_COUNT_LOSS_OPTIONS:-0.0 0.1}"

# segmentation is off by default -- set to "true" to also produce the seg-on
# results for every (base model x contrastive x CD x quad-count) combination.
# Segmentation only changes INFERENCE (see source/segmentation_utils.py's
# PySBDSegmenter), not training, so this does NOT train a separate seg-on
# model: each combination is trained once (as its seg-off scenario) and
# --also_eval_segmented then runs a second evaluation pass -- segmented --
# on that SAME checkpoint, writing results straight into the seg-on/ sibling
# folder (see gen_scl_nat_main.py's init_args/segmented_output_dir). So
# turning this on adds an eval pass, not a second training run.
RUN_SEGMENTATION=${RUN_SEGMENTATION:-true}

# same 5 seeds used across the other grid scripts in this repo (see
# configs/train_scl_all.sh), for consistency across experiments
SEEDS=(14298463 246773155 30288239 42511865 50995999 67584921 78912345 89012345 90123456 99568241)
# ablation grid: constrained decoding on/off (segmentation is a same-run
# eval-time byproduct via --also_eval_segmented, not a separate combination
# -- see RUN_SEGMENTATION above)
CD_OPTIONS=(false true)
N_PER_SEED=$(( ${#BASE_MODELS[@]} * ${#CONT_LOSS_OPTIONS[@]} * ${#QUAD_COUNT_LOSS_OPTIONS[@]} * ${#CD_OPTIONS[@]} ))
N_EXPECTED=$(( ${#SEEDS[@]} * N_PER_SEED ))

n_total=0
n_skipped=0
n_succeeded=0
n_failed=0
failed_runs=()

for base_model in "${BASE_MODELS[@]}"; do
    for cont_loss in "${CONT_LOSS_OPTIONS[@]}"; do
        for quad_count_loss in "${QUAD_COUNT_LOSS_OPTIONS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cd in "${CD_OPTIONS[@]}"; do
                    n_total=$((n_total + 1))

                    EXTRA_FLAGS=()
                    if [ "$cd" = "true" ]; then
                        EXTRA_FLAGS+=(--constrained_decoding)
                    fi
                    if [ "$RUN_SEGMENTATION" = "true" ]; then
                        EXTRA_FLAGS+=(--also_eval_segmented)
                    fi

                    echo ""
                    echo "=== [$n_total/$N_EXPECTED] GEN-SCL-NAT base_model=$base_model cont_loss=$cont_loss quad_count_loss=$quad_count_loss seed=$seed constrained_decoding=$cd also_eval_segmented=$RUN_SEGMENTATION ==="

                    run_log=$(mktemp)
                    python3 source/gen_scl_nat_main.py \
                        --task gen_scl_nat \
                        --absa_task "$ABSA_TASK" \
                        --do_train \
                        --do_direct_eval \
                        --scenario "$base_model" \
                        --dataset "$DATASET" \
                        --model_name_or_path t5-base \
                        --output_folder "$OUTPUT_FOLDER" \
                        --n_gpu 1 \
                        --accelerator gpu \
                        --train_batch_size 16 \
                        --eval_batch_size 16 \
                        --learning_rate 9e-5 \
                        --gradient_accumulation_steps 1 \
                        --num_train_epochs 45 \
                        --num_beams 5 \
                        --weight_decay 0.0 \
                        --seed "$seed" \
                        --cont_loss "$cont_loss" \
                        --cont_temp "$CONT_TEMP" \
                        --quad_count_loss "$quad_count_loss" \
                        --model_prefix "$MODEL_PREFIX" \
                        "${EXTRA_FLAGS[@]}" 2>&1 | tee "$run_log"
                    status=${PIPESTATUS[0]}

                    if grep -q '^\[RESUME\] Skipping' "$run_log"; then
                        n_skipped=$((n_skipped + 1))
                    elif [ $status -eq 0 ]; then
                        n_succeeded=$((n_succeeded + 1))
                    else
                        n_failed=$((n_failed + 1))
                        failed_runs+=("base_model=$base_model cont_loss=$cont_loss quad_count_loss=$quad_count_loss seed=$seed cd=$cd")
                        echo "[FAILED] base_model=$base_model cont_loss=$cont_loss quad_count_loss=$quad_count_loss seed=$seed constrained_decoding=$cd (exit code $status)" >&2
                    fi
                    rm -f "$run_log"
                done
            done
        done
    done
done

echo ""
echo "=== GEN-SCL-NAT grid search complete: $n_succeeded succeeded, $n_skipped skipped (already done), $n_failed failed, $n_total total ==="
if [ ${#failed_runs[@]} -gt 0 ]; then
    echo "Failed runs (re-run this script to retry them, completed ones will be skipped):"
    printf '  %s\n' "${failed_runs[@]}"
    exit 1
fi
