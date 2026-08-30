#!/bin/bash
# Grid search entry point for the GEN-SCL-NAT (--task gen_scl_nat) method on
# the drone forensics dataset: runs all 4 ablation combinations
# (baseline / +constrained-decoding / +segmentation / +both) across 5 seeds,
# i.e. 20 train+eval runs total.
#
# Resumable: source/gen_scl_nat_main.py's own output-folder check (keyed by
# dataset/scenario/task/absa_task/seed/cd-{on,off}/seg-{on,off}) skips any
# combination whose results-*.json already exists, so re-running this script
# after it stopped partway through (crash, preemption, Ctrl-C, ...) picks up
# where it left off instead of redoing completed runs.
#
# Usage:
#   bash configs/run_drone_gen_scl_nat.sh
#   DATASET=acos_drone_binary bash configs/run_drone_gen_scl_nat.sh
#   SEGMENTATION_MODEL_DIR=path/to/local/adfler/checkpoint bash configs/run_drone_gen_scl_nat.sh

set -uo pipefail

DATASET=${DATASET:-acos_drone_binary}
SCENARIO=${SCENARIO:-t5-base}
ABSA_TASK=${ABSA_TASK:-quad}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-train_outputs}
MODEL_PREFIX=${MODEL_PREFIX:-drone_gen_scl_nat}

# defaults to the published swardiantara/ADFLER-bert-base-cased checkpoint
# (source/gen_scl_nat_main.py's own default) when left unset
SEGMENTATION_MODEL_DIR=${SEGMENTATION_MODEL_DIR:-swardiantara/ADFLER-bert-base-cased}
SEGMENTATION_MODEL_TYPE=${SEGMENTATION_MODEL_TYPE:-bert}
SEGMENTATION_USE_CUDA=${SEGMENTATION_USE_CUDA:-true}

# same 5 seeds used across the other grid scripts in this repo (see
# configs/train_scl_all.sh), for consistency across experiments
SEEDS=(14298463 246773155 30288239 42511865 50995999)
# ablation grid: constrained decoding x segmentation, both on/off
CD_OPTIONS=(false true)
SEG_OPTIONS=(false true)

n_total=0
n_skipped=0
n_succeeded=0
n_failed=0
failed_runs=()

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
            echo "=== [$n_total/20] GEN-SCL-NAT seed=$seed constrained_decoding=$cd use_segmentation=$seg ==="

            run_log=$(mktemp)
            python3 source/gen_scl_nat_main.py \
                --task gen_scl_nat \
                --absa_task "$ABSA_TASK" \
                --do_train \
                --do_direct_eval \
                --scenario "$SCENARIO" \
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
                --cont_loss 0.05 \
                --cont_temp 0.25 \
                --model_prefix "$MODEL_PREFIX" \
                "${EXTRA_FLAGS[@]}" 2>&1 | tee "$run_log"
            status=${PIPESTATUS[0]}

            if grep -q '^\[RESUME\] Skipping' "$run_log"; then
                n_skipped=$((n_skipped + 1))
            elif [ $status -eq 0 ]; then
                n_succeeded=$((n_succeeded + 1))
            else
                n_failed=$((n_failed + 1))
                failed_runs+=("seed=$seed cd=$cd seg=$seg")
                echo "[FAILED] seed=$seed constrained_decoding=$cd use_segmentation=$seg (exit code $status)" >&2
            fi
            rm -f "$run_log"
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
