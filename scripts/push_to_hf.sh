#!/bin/bash
# Push model to HuggingFace Hub

set -e

# Configuration
CHECKPOINT_DIR=${1:-"./training_outputs/instruct_checkpoints"}
HF_MODEL_NAME=${2:-"your-username/instruct-lite"}
EXPORT_DIR="./hf"

echo "Exporting model to HuggingFace format..."
python scripts/export_to_hf.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --output "$EXPORT_DIR"

echo "Installing git-lfs if not already installed..."
git lfs install

echo "Logging into HuggingFace..."
huggingface-cli login

echo "Creating repository on HuggingFace Hub..."
huggingface-cli repo create "$HF_MODEL_NAME" --type model || true

echo "Pushing to HuggingFace Hub..."
cd "$EXPORT_DIR"
git init
git lfs track "*.bin"
git lfs track "*.pth"
git lfs track "*.safetensors"
git add .
git commit -m "Initial model upload"
git remote add origin "https://huggingface.co/$HF_MODEL_NAME"
git push -u origin main --force

echo "Model successfully pushed to: https://huggingface.co/$HF_MODEL_NAME"

