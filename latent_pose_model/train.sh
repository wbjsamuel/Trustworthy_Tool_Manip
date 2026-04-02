#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

torchrun --standalone --nnodes 1 --nproc-per-node 1 main.py fit \
    --config config/lam-stage-1.yaml \
    2>&1 | tee lam-stage-1.log
