#!/bin/bash
# Usage: ./run_experiment.sh "description"
# Commits train.py, runs it, extracts metrics, prints results
set -e
cd "$(dirname "$0")"
DESC="$1"
git add train.py
git commit -m "$DESC" --allow-empty
COMMIT=$(git rev-parse --short HEAD)
timeout 180 uv run train.py > run.log 2>&1 || true
ACC=$(grep "^val_accuracy:" run.log | awk '{print $2}' || echo "0.000000")
PARAMS=$(grep "^num_params:" run.log | awk '{print $2}' || echo "0")
TIME=$(grep "^training_seconds:" run.log | awk '{print $2}' || echo "0.0")
echo "RESULT: commit=$COMMIT acc=$ACC params=$PARAMS time=${TIME}s desc=$DESC"
echo -e "${COMMIT}\t${ACC}\t${PARAMS}\t${TIME}"
