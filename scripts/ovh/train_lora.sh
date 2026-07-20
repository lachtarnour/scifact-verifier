#!/usr/bin/env bash
set -euo pipefail

DATA_ARCHIVE_PATH="${DATA_ARCHIVE_PATH:-/data/scifact-retriever-data.tar.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-models/retriever/scifact-lora}"
REPORT_DIR="${REPORT_DIR:-reports/training/lora}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts}"
DATA_EXTRACT_DIR="${DATA_EXTRACT_DIR:-/workspace/run}"

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
FP16="${FP16:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
REQUIRE_GPU="${REQUIRE_GPU:-true}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
WANDB="${WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-scifact-retriever}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME:-scifact-lora}}"
WANDB_MODE="${WANDB_MODE:-online}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

required_files=(
  "data/processed/corpus.json"
  "data/processed/queries.json"
  "data/processed/qrels_train.json"
  "data/processed/qrels_test.json"
  "data/processed/fine_tuning/retriever/train.jsonl"
  "data/processed/fine_tuning/retriever/validation.jsonl"
  "data/processed/fine_tuning/retriever/test.jsonl"
)

has_dataset() {
  local file
  for file in "${required_files[@]}"; do
    [[ -f "$file" ]] || return 1
  done
}

extract_dataset() {
  local archive="/tmp/scifact-retriever-data.tar.gz"

  if [[ ! -f "$DATA_ARCHIVE_PATH" ]]; then
    echo "Dataset archive not found: $DATA_ARCHIVE_PATH"
    exit 1
  fi

  cp "$DATA_ARCHIVE_PATH" "$archive"
  mkdir -p "$DATA_EXTRACT_DIR"
  tar \
    --no-same-owner \
    --no-same-permissions \
    --warning=no-unknown-keyword \
    --exclude="._*" \
    --exclude=".DS_Store" \
    -xzf "$archive" \
    -C "$DATA_EXTRACT_DIR"
}

if ! has_dataset; then
  extract_dataset
fi

if ! has_dataset; then
  echo "Dataset archive was extracted, but required files are still missing."
  printf 'Missing or expected files:\n'
  printf -- '- %s\n' "${required_files[@]}"
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$ARTIFACT_DIR" /cache/huggingface

python - <<'PY'
import os
import sys
import torch

if not torch.cuda.is_available():
    if os.environ.get("REQUIRE_GPU", "true") == "true":
        sys.exit("CUDA is not available. Run Docker with --gpus all or set REQUIRE_GPU=false.")
    sys.exit(0)

device = torch.cuda.current_device()
gb = 1024**3
print(f"gpu ram allocated: {torch.cuda.memory_allocated(device) / gb:.2f} GiB")
PY

train_args=(
  -m src.finetuning.retriever.train_lora
  --train-file data/processed/fine_tuning/retriever/train.jsonl
  --validation-file data/processed/fine_tuning/retriever/validation.jsonl
  --output-dir "$OUTPUT_DIR"
  --log-file "$REPORT_DIR/loss.jsonl"
  --validation-log-file "$REPORT_DIR/validation_metrics.jsonl"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
)

if [[ "$FP16" == "true" ]]; then
  train_args+=(--fp16)
fi

if [[ "$WANDB" == "true" ]]; then
  train_args+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$WANDB_RUN_NAME"
    --wandb-mode "$WANDB_MODE"
  )
fi

python "${train_args[@]}"

if [[ "$RUN_EVAL" == "true" ]]; then
  python -m src.finetuning.retriever.evaluate \
    --lora-adapter "$OUTPUT_DIR" \
    --split test \
    --device "$EVAL_DEVICE" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --no-progress | tee "$REPORT_DIR/eval_test.txt"
fi

model_archive="$ARTIFACT_DIR/scifact-lora.tar.gz"
tar -czf "$model_archive" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"
tar -czf "$ARTIFACT_DIR/lora-reports.tar.gz" -C "$(dirname "$REPORT_DIR")" "$(basename "$REPORT_DIR")"

echo "Training complete."
echo "Model: $OUTPUT_DIR"
echo "Loss log: $REPORT_DIR/loss.jsonl"
echo "Artifacts: $ARTIFACT_DIR"
