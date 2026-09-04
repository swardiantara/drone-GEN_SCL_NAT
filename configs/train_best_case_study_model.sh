#!/bin/bash
# Trains and saves the single best-performing scenario from the grid search
# (acos_drone_binary, t5-base, paraphrase, contrastive on, constrained
# decoding on, segmentation off, seed=42511865 -- see analysis/aggregate/
# multiset-PRF.xlsx), so its checkpoint can be reused for case-study
# inference (source/case_study_inference.py) instead of only ever being
# scored and discarded like the rest of the grid.
#
# --overwrite bypasses the resume check: this exact scenario/seed already
# has a results-*.json from the grid search (configs/run_drone_paraphrase.sh),
# but without --save_model, so its weights were never kept. This reruns it
# with --save_model so the HF checkpoint (config.json/pytorch_model.bin/
# tokenizer files) lands in its output folder, then copies that into
# best-model/ at the repo root.
#
# Usage (from the repo root):
#   bash configs/train_best_case_study_model.sh

set -euo pipefail

DATASET=${DATASET:-acos_drone_binary}
ABSA_TASK=${ABSA_TASK:-quad}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-train_outputs}
MODEL_PREFIX=${MODEL_PREFIX:-drone_paraphrase}
BASE_MODEL=${BASE_MODEL:-t5-base}
SEED=${SEED:-42511865}
CONT_LOSS=${CONT_LOSS:-0.05}
CONT_TEMP=${CONT_TEMP:-0.25}
BEST_MODEL_DIR=${BEST_MODEL_DIR:-best-model}

python3 source/gen_scl_nat_main.py \
    --task asqp \
    --absa_task "$ABSA_TASK" \
    --do_train \
    --do_direct_eval \
    --scenario "$BASE_MODEL" \
    --dataset "$DATASET" \
    --model_name_or_path "$BASE_MODEL" \
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
    --seed "$SEED" \
    --cont_loss "$CONT_LOSS" \
    --cont_temp "$CONT_TEMP" \
    --model_prefix "$MODEL_PREFIX" \
    --constrained_decoding \
    --save_model \
    --overwrite

RUN_DIR="$OUTPUT_FOLDER/$DATASET/$BASE_MODEL/asqp/$ABSA_TASK/cont-on/cd-on/seg-off/$SEED"
if [ ! -f "$RUN_DIR/config.json" ]; then
    echo "[FAILED] expected a saved HF checkpoint at $RUN_DIR (config.json missing)" >&2
    exit 1
fi

rm -rf "$BEST_MODEL_DIR"
mkdir -p "$BEST_MODEL_DIR"
# copy every save_pretrained() artifact (model weights + tokenizer files --
# exact filenames vary by transformers version), skipping this run folder's
# own bookkeeping (args.json / results-*.json / test_results.txt), which
# aren't part of the HF checkpoint and would just be confusing alongside it
for f in "$RUN_DIR"/*; do
    base="$(basename "$f")"
    case "$base" in
        args.json|results-*.json|test_results.txt) continue ;;
    esac
    [ -f "$f" ] && cp "$f" "$BEST_MODEL_DIR"/
done

echo ""
echo "=== Saved best-performing checkpoint to $BEST_MODEL_DIR/ (from $RUN_DIR) ==="
ls -la "$BEST_MODEL_DIR"
