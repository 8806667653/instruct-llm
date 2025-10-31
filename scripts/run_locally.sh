#!/bin/bash
# Quick local run script

set -e

# Default values
COMMAND=${1:-"train"}
DATA_PATH=${2:-"data/raw/alpaca_sample.jsonl"}

echo "Running instruct-lite: $COMMAND"

case $COMMAND in
  train)
    echo "Training model..."
    python -m src.utils.cli train \
      --data "$DATA_PATH" \
      --model gpt2 \
      --epochs 3 \
      --batch-size 4 \
      --lr 5e-5 \
      --output ./training_outputs/instruct_checkpoints
    ;;
    
  test)
    MODEL_PATH=${2:-"gpt2"}
    echo "Testing model: $MODEL_PATH"
    python -m src.utils.cli test \
      --model "$MODEL_PATH" \
      --device cuda
    ;;
    
  export)
    CHECKPOINT=${2:-"./training_outputs/instruct_checkpoints"}
    OUTPUT=${3:-"./hf"}
    echo "Exporting model..."
    python scripts/export_to_hf.py \
      --checkpoint "$CHECKPOINT" \
      --output "$OUTPUT"
    ;;
    
  *)
    echo "Usage: $0 {train|test|export} [args...]"
    echo ""
    echo "Examples:"
    echo "  $0 train data/raw/alpaca_sample.jsonl"
    echo "  $0 test ./training_outputs/instruct_checkpoints"
    echo "  $0 export ./training_outputs/instruct_checkpoints ./hf"
    exit 1
    ;;
esac

echo "Done!"

