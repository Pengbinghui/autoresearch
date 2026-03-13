#!/bin/bash
# Batch experiment runner. Takes experiment description and train.py content from stdin.
# Usage: echo "TRAIN_PY_CONTENT" | ./run_batch.sh "BEST_COMMIT" "description"
set -e
cd "$(dirname "$0")"
BEST_COMMIT="$1"
DESC="$2"

# Write train.py from stdin
cat > train.py

# Commit
git add train.py
git commit -m "$DESC" -q 2>/dev/null || true
COMMIT=$(git rev-parse --short HEAD)

# Run with timeout
timeout 200 uv run train.py > run.log 2>&1 || true

# Extract metrics
ACC=$(grep "^val_accuracy:" run.log 2>/dev/null | awk '{print $2}')
PARAMS=$(grep "^num_params:" run.log 2>/dev/null | awk '{print $2}')
TIME=$(grep "^training_seconds:" run.log 2>/dev/null | awk '{print $2}')

# Defaults for crashes
ACC=${ACC:-0.000000}
PARAMS=${PARAMS:-0}
TIME=${TIME:-0.0}

# Determine status
ACC_INT=$(echo "$ACC" | awk '{printf "%d", $1 * 10000}')
if [ "$ACC_INT" -ge 9800 ] 2>/dev/null; then
    STATUS="keep"
elif [ "$ACC_INT" -eq 0 ] 2>/dev/null; then
    STATUS="crash"
else
    STATUS="discard"
fi

# Log
echo -e "${COMMIT}\t${ACC}\t${PARAMS}\t${TIME}\t${STATUS}\t${DESC}" >> results.tsv

# Print result
echo "${STATUS}|${ACC}|${PARAMS}|${TIME}|${COMMIT}"

# Reset if not keeping
if [ "$STATUS" != "keep" ]; then
    git reset --hard "$BEST_COMMIT" -q 2>/dev/null
fi
