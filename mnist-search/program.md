# mnist-search

Autonomous MNIST classifier research on CPU.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/mnist-<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/mnist-<tag>` from current master.
3. **Read the in-scope files**:
   - `mnist-search/prepare.py` — fixed constants, data loading, evaluation. Do not modify.
   - `mnist-search/train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/mnist-search/` contains MNIST data. If not, tell the human to run `cd mnist-search && uv run prepare.py`.
5. **Initialize results.tsv**: Create `mnist-search/results.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs on CPU. Launch: `cd mnist-search && uv run train.py`

**Time budget per run:**
- **Goal A**: Fixed 60 seconds (use the full budget to maximize accuracy).
- **Goals B and C**: Upper bound of 120 seconds. The agent should override `TIME_BUDGET` in `train.py` (e.g. `TIME_BUDGET = 120` after the import) and add early stopping so training halts once the accuracy target is met or accuracy plateaus. Most runs should finish well under 120s.

**What you CAN do:**
- Modify `mnist-search/train.py` — everything is fair game: model architecture (MLP, CNN, etc.), optimizer, hyperparameters, regularization, data augmentation, learning rate schedules, batch size.

**What you CANNOT do:**
- Modify `mnist-search/prepare.py`. It contains the fixed evaluation and data loading.
- Install new packages or add dependencies.

**Experiment goals** — the user specifies which goal to pursue when starting a session:

- **Goal A — Highest accuracy**: Maximize `val_accuracy`. Keep if accuracy improves. This is the default.
- **Goal B — Smallest model**: Achieve ≥98% accuracy with the fewest `num_params`. The agent should add early stopping so training halts once accuracy plateaus (no point waiting for the full budget). Keep if `num_params` decreased while `val_accuracy >= 0.98`.
- **Goal C — Fastest convergence**: Reach ≥98% accuracy in the fewest `training_seconds`. The agent should add periodic evaluation and early stopping in `train.py` so training halts as soon as the threshold is met. Keep if `training_seconds` decreased while `val_accuracy >= 0.98`.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Always establish the baseline first by running train.py unmodified.

## Output format

```
---
val_accuracy:     0.972000
training_seconds: 60.0
total_seconds:    63.2
total_epochs:     5
num_steps:        4690
num_params:       203530
```

Extract the metric: `grep "^val_accuracy:" run.log`

## Logging results

Log to `mnist-search/results.tsv` (tab-separated, NOT comma-separated). Do NOT commit this file.

```
commit	val_accuracy	num_params	training_seconds	status	description
a1b2c3d	0.972000	203530	60.0	keep	baseline
b2c3d4e	0.981500	415242	58.2	keep	increase hidden to 512
c3d4e5f	0.965000	203530	60.0	discard	switch to SGD without momentum
d4e5f6g	0.000000	0	0.0	crash	typo in forward pass
```

Columns: commit (7 chars), val_accuracy (0.000000 for crashes), num_params (0 for crashes), training_seconds (0.0 for crashes), status (keep/discard/crash), description.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mnist-mar9`).

Run for approximately **60 experiments (~1 hour)**, or until manually stopped.

LOOP (up to ~60 iterations):

1. Look at the git state: current branch/commit
2. Edit `mnist-search/train.py` with an experimental idea
3. git commit
4. Run: `cd mnist-search && uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_accuracy:\|^num_params:\|^training_seconds:" run.log`
6. If grep empty → crashed. `tail -n 50 run.log` for traceback, attempt fix.
7. Record results in results.tsv
8. Compare against the current best using the active goal's keep/discard logic:
   - **Goal A**: keep if `val_accuracy` improved (higher)
   - **Goal B**: keep if `num_params` decreased AND `val_accuracy >= 0.98`
   - **Goal C**: keep if `training_seconds` decreased AND `val_accuracy >= 0.98`
9. If the goal's condition is met → keep the commit
10. Otherwise → `git reset` back

**Timeout**: Goal A runs take ~60s + overhead. Goals B/C runs have a 120s upper bound but should finish sooner with early stopping. If any run exceeds 3 minutes, kill it and treat as failure.

**Crashes**: If it's a typo or easy fix, fix and re-run. If fundamentally broken, log "crash" and move on.

**Keep going**: Once the loop begins, do NOT pause to ask the human. Keep running until ~60 experiments are complete or you are manually stopped. If you run out of ideas, think harder — try CNNs, batch normalization, learning rate schedules, data augmentation, dropout, different optimizers, weight initialization, etc.
