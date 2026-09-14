#!/bin/bash
# Runs the case-study inference pipeline (source/case_study_inference.py)
# over the seven real-world flight-log evidence files under
# evidence/parsed/*.xlsx, using the best-performing checkpoint
# (configs/train_best_case_study_model.sh -> best-model/), once with
# sentence segmentation off and once with it on -- so their predictions can
# be compared/scored against each other (results land in separate folders,
# case_study/predictions/seg-off/ and case_study/predictions/seg-on/, since
# --output_dir defaults based on --segment_messages -- see
# source/case_study_inference.py).
#
# best-model/ was trained with --constrained_decoding on (see
# configs/train_best_case_study_model.sh), which is also
# source/case_study_inference.py's default, so no extra flag is needed for
# that here.
#
# Usage (from the repo root):
#   bash configs/run_case_study_inference.sh
#   MODEL_DIR=best-model EVIDENCE_DIR=evidence/parsed bash configs/run_case_study_inference.sh
#   USE_CUDA=true bash configs/run_case_study_inference.sh

set -euo pipefail

EVIDENCE_DIR=${EVIDENCE_DIR:-evidence/parsed}
MODEL_DIR=${MODEL_DIR:-best-model}
DATASET=${DATASET:-acos_drone_binary}
BATCH_SIZE=${BATCH_SIZE:-16}
USE_CUDA=${USE_CUDA:-false}

EXTRA_FLAGS=()
if [ "$USE_CUDA" = "true" ]; then
    EXTRA_FLAGS+=(--use_cuda)
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[FAILED] no saved HF checkpoint found at $MODEL_DIR (config.json missing) -- run configs/train_best_case_study_model.sh first" >&2
    exit 1
fi

echo ""
echo "=== Case-study inference: segmentation OFF ==="
python3 source/case_study_inference.py \
    --evidence_dir "$EVIDENCE_DIR" \
    --model_dir "$MODEL_DIR" \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "=== Case-study inference: segmentation ON (PySBD) ==="
python3 source/case_study_inference.py \
    --evidence_dir "$EVIDENCE_DIR" \
    --model_dir "$MODEL_DIR" \
    --dataset "$DATASET" \
    --batch_size "$BATCH_SIZE" \
    --segment_messages \
    "${EXTRA_FLAGS[@]}"

echo ""
echo "=== Done: predictions in case_study/predictions/seg-off/ and case_study/predictions/seg-on/ ==="
