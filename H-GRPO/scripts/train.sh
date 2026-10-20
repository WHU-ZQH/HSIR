#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?MedQA or GSM8K}"
MODEL="${2:?absolute model path}"
DATA="${3:?absolute JSON data path}"
OUT="${4:?output directory}"
[[ "$TASK" == MedQA || "$TASK" == GSM8K ]] || { echo "Unknown task: $TASK" >&2; exit 2; }
if [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then echo "Use a fresh output directory" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
accelerate launch --config_file "$ROOT/recipes/zero3.yaml" \
  --num_processes "${NUM_PROCESSES:-7}" "$ROOT/open_r1/grpo.py" \
  --model_name_or_path "$MODEL" --dataset_name "$DATA" --output_dir "$OUT" \
  --learning_rate "${LEARNING_RATE:-1e-6}" --config "$ROOT/recipes/H-GRPO_${TASK}.yaml" \
  2>&1 | tee "$OUT/train.log"
