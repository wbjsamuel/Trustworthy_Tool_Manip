#!/bin/bash

# This script runs the training for the Stage 1 model.
# Ensure you have installed the required dependencies from requirements.txt

echo "Starting Stage 1 training..."
python -m stage1.train
echo "Training finished."
