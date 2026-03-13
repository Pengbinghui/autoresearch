#!/usr/bin/env python3
"""Goal D2: Minimum linear projection dimension d.

Linear(784, d) -> arbitrary NN -> 10 classes.
Minimize d while val_accuracy >= 0.95.

Strategy: binary search on d, with adaptive architecture/hyperparameter tuning.
"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import get_data, make_batches, evaluate_accuracy

TARGET_ACC = 0.95
MAX_TRAIN_TIME = 180  # seconds per trial


def make_model(d, hidden_sizes, dropout=0.0):
    """Build Linear(784,d) -> NN(d -> 10)."""
    layers = [nn.Flatten(), nn.Linear(784, d)]
    prev = d
    for h in hidden_sizes:
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev, h))
        prev = h
    layers.append(nn.ReLU())
    layers.append(nn.Linear(prev, 10))
    return nn.Sequential(*layers)


def train_and_eval(d, hidden_sizes, lr=1e-3, bs=64, epochs=50, dropout=0.0,
                   weight_decay=0.0, verbose=False):
    """Train a model with projection dim d. Returns best val accuracy."""
    torch.manual_seed(42)
    model = make_model(d, hidden_sizes, dropout=dropout)
    num_params = sum(p.numel() for p in model.parameters())

    train_images, train_labels, _, _ = get_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    best_acc = 0.0
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        for x, y in make_batches(train_images, train_labels, bs):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model)
        model.train()
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - t_start
        if verbose:
            print(f"  epoch {epoch} | acc={acc:.4f} | best={best_acc:.4f} | {elapsed:.1f}s")

        if no_improve >= 5:
            break
        if elapsed > MAX_TRAIN_TIME:
            break

    elapsed = time.time() - t_start
    return best_acc, num_params, elapsed


def try_d(d, configs=None):
    """Try dimension d with multiple configurations. Returns best accuracy achieved."""
    if configs is None:
        configs = [
            # (hidden_sizes, lr, bs, dropout, wd)
            ([128, 64], 1e-3, 64, 0.0, 0.0),
            ([256, 128], 1e-3, 64, 0.0, 0.0),
            ([64, 32], 1e-3, 64, 0.0, 0.0),
            ([128], 1e-3, 64, 0.0, 0.0),
        ]

    best_acc = 0.0
    best_config = None
    for hidden, lr, bs, dropout, wd in configs:
        acc, params, elapsed = train_and_eval(
            d, hidden, lr=lr, bs=bs, dropout=dropout, weight_decay=wd
        )
        print(f"  d={d} hidden={hidden} lr={lr} bs={bs} do={dropout} wd={wd} -> acc={acc:.4f} params={params} time={elapsed:.1f}s")
        if acc > best_acc:
            best_acc = acc
            best_config = (hidden, lr, bs, dropout, wd)
        if acc >= TARGET_ACC:
            break  # no need to try more configs

    return best_acc, best_config


def run():
    print(f"=== Goal D2: Minimum linear projection dimension ===")
    print(f"Target: {TARGET_ACC*100:.0f}% accuracy")
    print()

    results = {}  # d -> (best_acc, best_config)

    # Phase 1: Coarse binary search
    print("--- Phase 1: Coarse search ---")
    lo, hi = 1, 784
    # Quick check: does d=784 work?
    acc, cfg = try_d(784, [([128, 64], 1e-3, 64, 0.0, 0.0)])
    results[784] = (acc, cfg)
    print(f"d=784: acc={acc:.4f} {'PASS' if acc >= TARGET_ACC else 'FAIL'}")

    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        print(f"\nTrying d={mid} (range [{lo}, {hi}])")
        acc, cfg = try_d(mid)
        results[mid] = (acc, cfg)
        if acc >= TARGET_ACC:
            hi = mid
            print(f"  d={mid}: PASS ({acc:.4f}) -> search [{lo}, {hi}]")
        else:
            lo = mid + 1
            print(f"  d={mid}: FAIL ({acc:.4f}) -> search [{lo}, {hi}]")

    # Phase 2: Fine-tune around the boundary
    print(f"\n--- Phase 2: Fine-tune around d={lo} ---")
    boundary = lo

    # Try d-1, d-2, d-3 with more aggressive training
    aggressive_configs = [
        ([256, 128], 1e-3, 64, 0.0, 0.0),
        ([256, 128], 5e-4, 64, 0.0, 0.0),
        ([512, 256], 1e-3, 64, 0.0, 0.0),
        ([256, 128, 64], 1e-3, 64, 0.0, 0.0),
        ([128, 64], 1e-3, 32, 0.0, 0.0),
        ([256, 128], 1e-3, 64, 0.1, 0.0),
        ([256, 128], 1e-3, 64, 0.0, 1e-5),
        ([256, 128], 2e-3, 128, 0.0, 0.0),
    ]

    for offset in range(1, 6):
        d_try = boundary - offset
        if d_try < 1:
            break
        if d_try in results and results[d_try][0] >= TARGET_ACC:
            continue
        print(f"\nAggressive try d={d_try}")
        acc, cfg = try_d(d_try, aggressive_configs)
        results[d_try] = (acc, cfg)
        if acc >= TARGET_ACC:
            boundary = d_try
            print(f"  d={d_try}: PASS! New boundary = {boundary}")
        else:
            print(f"  d={d_try}: FAIL ({acc:.4f})")

    # Phase 3: Squeeze even harder at boundary-1 with many configs
    print(f"\n--- Phase 3: Squeeze at d={boundary-1} ---")
    d_squeeze = boundary - 1
    if d_squeeze >= 1:
        squeeze_configs = [
            ([512, 256, 128], 1e-3, 64, 0.0, 0.0),
            ([512, 256, 128], 5e-4, 32, 0.0, 0.0),
            ([1024, 512], 1e-3, 64, 0.0, 0.0),
            ([256, 256, 128], 1e-3, 64, 0.1, 0.0),
            ([512, 256], 5e-4, 64, 0.0, 1e-5),
            ([256, 128], 3e-4, 32, 0.0, 0.0),
            ([512, 256, 128], 1e-3, 128, 0.0, 0.0),
            ([256, 128], 1e-3, 64, 0.2, 1e-5),
        ]
        print(f"\nHard squeeze d={d_squeeze}")
        acc, cfg = try_d(d_squeeze, squeeze_configs)
        results[d_squeeze] = (acc, cfg)
        if acc >= TARGET_ACC:
            boundary = d_squeeze
            print(f"  d={d_squeeze}: PASS! New boundary = {boundary}")

            # Keep going down
            for offset in range(1, 4):
                d_next = boundary - offset
                if d_next < 1:
                    break
                print(f"\nHard squeeze d={d_next}")
                acc, cfg = try_d(d_next, squeeze_configs)
                results[d_next] = (acc, cfg)
                if acc >= TARGET_ACC:
                    boundary = d_next
                    print(f"  d={d_next}: PASS! New boundary = {boundary}")
                else:
                    print(f"  d={d_next}: FAIL ({acc:.4f})")
                    break

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: minimum d = {boundary}")
    passing = {d: r for d, r in results.items() if r[0] >= TARGET_ACC}
    if passing:
        best_d = min(passing.keys())
        best_acc, best_cfg = passing[best_d]
        print(f"Best passing: d={best_d}, acc={best_acc:.4f}, config={best_cfg}")

    print(f"\nAll results:")
    for d in sorted(results.keys()):
        acc, cfg = results[d]
        mark = "PASS" if acc >= TARGET_ACC else "FAIL"
        print(f"  d={d:4d}: acc={acc:.4f} [{mark}]  config={cfg}")


if __name__ == "__main__":
    run()
