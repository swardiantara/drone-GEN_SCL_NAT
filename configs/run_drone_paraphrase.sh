#!/bin/bash
# Reproducible entry point for the PARAPHRASE (ABSA-QUAD, --task asqp) method
# on the drone forensics dataset, with optional ablation switches for
# constrained decoding and inference-time sentence segmentation.
#
# Usage:
#   bash configs/run_drone_paraphrase.sh
#   USE_CONSTRAINED_DECODING=true bash configs/run_drone_paraphrase.sh
#   USE_SEGMENTATION=true SEGMENTATION_MODEL_DIR=models/lognexus_drone bash configs/run_drone_paraphrase.sh
#   DATASET=acos_drone_binary bash configs/run_drone_paraphrase.sh

set -e

DATASET=${DATASET:-acos_drone_multi}
SCENARIO=${SCENARIO:-t5-base}
ABSA_TASK=${ABSA_TASK:-quad}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-train_outputs}
MODEL_PREFIX=${MODEL_PREFIX:-drone_paraphrase}
SEED=${SEED:-42}

# --- ablation switches --------------------------------------------------
# Set to "true" to enable each option; both are off by default.
USE_CONSTRAINED_DECODING=${USE_CONSTRAINED_DECODING:-false}
USE_SEGMENTATION=${USE_SEGMENTATION:-false}
SEGMENTATION_MODEL_DIR=${SEGMENTATION_MODEL_DIR:-}
SEGMENTATION_USE_CUDA=${SEGMENTATION_USE_CUDA:-false}
# -------------------------------------------------------------------------

EXTRA_FLAGS=()
if [ "$USE_CONSTRAINED_DECODING" = "true" ]; then
    EXTRA_FLAGS+=(--constrained_decoding)
fi
if [ "$USE_SEGMENTATION" = "true" ]; then
    if [ -z "$SEGMENTATION_MODEL_DIR" ]; then
        echo "USE_SEGMENTATION=true requires SEGMENTATION_MODEL_DIR to be set" >&2
        exit 1
    fi
    EXTRA_FLAGS+=(--use_segmentation --segmentation_model_dir "$SEGMENTATION_MODEL_DIR")
    if [ "$SEGMENTATION_USE_CUDA" = "true" ]; then
        EXTRA_FLAGS+=(--segmentation_use_cuda)
    fi
fi

python3 source/gen_scl_nat_main.py \
   --task asqp \
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
   --seed "$SEED" \
   --cont_loss 0.05 \
   --cont_temp 0.25 \
   --model_prefix "$MODEL_PREFIX" \
   "${EXTRA_FLAGS[@]}"
