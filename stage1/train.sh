#!/bin/bash

# This script runs the training for the Stage 1 model.
# Ensure you have installed the required dependencies from requirements.txt

export HF_HOME="./cache"
export HF_TOKEN="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"
export TORCH_HOME="./torch_cache"
export WANDB_API_KEY="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"

echo "Starting Stage 1 training..."
python3 -m stage1.train "$@"
echo "Training finished."
