#!/bin/bash
# Grid search entry point for the PARAPHRASE (ABSA-QUAD, --task asqp) method
# on the drone forensics dataset: one invocation runs every scenario --
# base model x contrastive-loss on/off x constrained-decoding on/off --
# across 5 seeds. Default grid: 2 base models x 2 contrastive x 2 CD x 5
# seeds = 40 train+eval runs. Segmentation is off by default (dropped from
# the ablation scope); set RUN_SEGMENTATION=true to add it back in.
#
# Resumable: source/gen_scl_nat_main.py's own output-folder check (keyed by
# dataset/scenario/task/absa_task/cont-{on,off}/cd-{on,off}/seg-{on,off}/seed)
# skips any combination whose results-*.json already exists, so re-running
# this script after it stopped partway through (crash, preemption, Ctrl-C,
# ...) picks up where it left off instead of redoing completed runs. If
# you've deleted train_outputs/ to start clean (e.g. after a dataset fix),
# every combination runs fresh.
#
# Usage:
#   bash configs/run_drone_paraphrase.sh
#   DATASET=acos_drone_binary bash configs/run_drone_paraphrase.sh
#   BASE_MODELS="t5-base flan-t5-base" bash configs/run_drone_paraphrase.sh
#   CONT_LOSS_OPTIONS="0.0 0.05" bash configs/run_drone_paraphrase.sh
#   RUN_SEGMENTATION=true SEGMENTATION_MODEL_DIR=path/to/local/adfler/checkpoint bash configs/run_drone_paraphrase.sh

set -uo pipefail

DATASET=${DATASET:-acos_drone_binary}
ABSA_TASK=${ABSA_TASK:-quad}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-train_outputs}
MODEL_PREFIX=${MODEL_PREFIX:-drone_paraphrase}

# base-model scenarios to sweep (space-separated; each becomes --scenario,
# see source/gen_scl_nat_main.py's get_seq2seq_model() for the full list of
# recognized values, e.g. t5-base, flan-t5, flan-t5-large, bert2gpt2, ...)
read -ra BASE_MODELS <<< "${BASE_MODELS:-t5-base flan-t5-base}"

# --cont_loss/--cont_temp: T5FineTuner._step always computes the SCL
# auxiliary loss, scaled by --cont_loss -- 0.0 makes its contribution exactly
# zero (see source/losses.py's SupConLoss), i.e. contrastive learning OFF.
# gen_scl_nat_main.py's output folder already encodes this as cont-{on,off}
# (derived from --cont_loss > 0), so each cont_loss value lands in its own
# folder and is resumable/isolated independently.
read -ra CONT_LOSS_OPTIONS <<< "${CONT_LOSS_OPTIONS:-0.0 0.05}"
CONT_TEMP=${CONT_TEMP:-0.25}

# defaults to the published swardiantara/ADFLER-xlnet-base-cased checkpoint
# (source/gen_scl_nat_main.py's own default) when left unset
SEGMENTATION_MODEL_DIR=${SEGMENTATION_MODEL_DIR:-swardiantara/ADFLER-xlnet-base-cased}
SEGMENTATION_MODEL_TYPE=${SEGMENTATION_MODEL_TYPE:-xlnet}
SEGMENTATION_USE_CUDA=${SEGMENTATION_USE_CUDA:-true}
# segmentation is off by default -- set to "true" to add the seg=on half of
# the grid back in for every (base model x contrastive x CD) combination
RUN_SEGMENTATION=${RUN_SEGMENTATION:-false}

# same 5 seeds used across the other grid scripts in this repo (see
# configs/train_scl_all.sh), for consistency across experiments
SEEDS=(14298463 246773155 30288239 42511865 50995999)
# ablation grid: constrained decoding x segmentation, both on/off
CD_OPTIONS=(false true)
if [ "$RUN_SEGMENTATION" = "true" ]; then
    SEG_OPTIONS=(false true)
else
    SEG_OPTIONS=(false)
fi
N_PER_SEED=$(( ${#BASE_MODELS[@]} * ${#CONT_LOSS_OPTIONS[@]} * ${#CD_OPTIONS[@]} * ${#SEG_OPTIONS[@]} ))
N_EXPECTED=$(( ${#SEEDS[@]} * N_PER_SEED ))

n_total=0
n_skipped=0
n_succeeded=0
n_failed=0
failed_runs=()

for base_model in "${BASE_MODELS[@]}"; do
    for cont_loss in "${CONT_LOSS_OPTIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for cd in "${CD_OPTIONS[@]}"; do
                for seg in "${SEG_OPTIONS[@]}"; do
                    n_total=$((n_total + 1))

                    EXTRA_FLAGS=()
                    if [ "$cd" = "true" ]; then
                        EXTRA_FLAGS+=(--constrained_decoding)
                    fi
                    if [ "$seg" = "true" ]; then
                        EXTRA_FLAGS+=(--use_segmentation --segmentation_model_type "$SEGMENTATION_MODEL_TYPE")
                        if [ -n "$SEGMENTATION_MODEL_DIR" ]; then
                            EXTRA_FLAGS+=(--segmentation_model_dir "$SEGMENTATION_MODEL_DIR")
                        fi
                        if [ "$SEGMENTATION_USE_CUDA" = "true" ]; then
                            EXTRA_FLAGS+=(--segmentation_use_cuda)
                        fi
                    fi

                    echo ""
                    echo "=== [$n_total/$N_EXPECTED] PARAPHRASE base_model=$base_model cont_loss=$cont_loss seed=$seed constrained_decoding=$cd use_segmentation=$seg ==="

                    run_log=$(mktemp)
                    python3 source/gen_scl_nat_main.py \
                        --task asqp \
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
                        --model_prefix "$MODEL_PREFIX" \
                        "${EXTRA_FLAGS[@]}" 2>&1 | tee "$run_log"
                    status=${PIPESTATUS[0]}

                    if grep -q '^\[RESUME\] Skipping' "$run_log"; then
                        n_skipped=$((n_skipped + 1))
                    elif [ $status -eq 0 ]; then
                        n_succeeded=$((n_succeeded + 1))
                    else
                        n_failed=$((n_failed + 1))
                        failed_runs+=("base_model=$base_model cont_loss=$cont_loss seed=$seed cd=$cd seg=$seg")
                        echo "[FAILED] base_model=$base_model cont_loss=$cont_loss seed=$seed constrained_decoding=$cd use_segmentation=$seg (exit code $status)" >&2
                    fi
                    rm -f "$run_log"
                done
            done
        done
    done
done

echo ""
echo "=== PARAPHRASE grid search complete: $n_succeeded succeeded, $n_skipped skipped (already done), $n_failed failed, $n_total total ==="
if [ ${#failed_runs[@]} -gt 0 ]; then
    echo "Failed runs (re-run this script to retry them, completed ones will be skipped):"
    printf '  %s\n' "${failed_runs[@]}"
    exit 1
fi
