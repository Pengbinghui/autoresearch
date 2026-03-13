#!/usr/bin/env python3
"""Goal D1: Fewest input positions to reach >= 95% accuracy.

Select as few pixel positions as possible from the 28x28 input.
Model only sees values at those positions.

Strategy:
1. Rank pixels by importance (variance * mutual-information proxy)
2. Binary search on number of pixels needed
3. Greedy refinement: swap/drop pixels to minimize count
"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from prepare import get_data, make_batches, evaluate_accuracy

TARGET_ACC = 0.95
MAX_TRAIN_TIME = 180  # seconds per trial


class PixelSelectModel(nn.Module):
    """Model that selects specific pixel positions from input."""
    def __init__(self, positions, hidden_sizes):
        super().__init__()
        self.positions = positions  # list of (row, col)
        n_pixels = len(positions)
        # Register positions as buffer for forward pass
        rows = torch.tensor([p[0] for p in positions], dtype=torch.long)
        cols = torch.tensor([p[1] for p in positions], dtype=torch.long)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)

        layers = []
        prev = n_pixels
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 10))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        selected = x[:, 0, self.rows, self.cols]  # [B, n_pixels]
        return self.classifier(selected)


def train_and_eval(positions, hidden_sizes, lr=1e-3, bs=64, epochs=50,
                   weight_decay=0.0, verbose=False):
    """Train model with given pixel positions. Returns best val accuracy."""
    torch.manual_seed(42)
    model = PixelSelectModel(positions, hidden_sizes)
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


def rank_pixels():
    """Rank all 784 pixels by importance using variance and per-class discriminability."""
    train_images, train_labels, _, _ = get_data()
    # Flatten to [N, 784]
    flat = train_images.view(train_images.size(0), -1)  # [60000, 784]

    # Method 1: Pixel variance (higher variance = more informative)
    variance = flat.var(dim=0)  # [784]

    # Method 2: Per-class mean separation (F-statistic proxy)
    grand_mean = flat.mean(dim=0)  # [784]
    between_var = torch.zeros(784)
    for c in range(10):
        mask = train_labels == c
        class_mean = flat[mask].mean(dim=0)
        between_var += mask.sum().float() * (class_mean - grand_mean) ** 2
    between_var /= train_images.size(0)

    # Combined score
    score = between_var * variance
    # Convert to (row, col) positions
    ranked_indices = score.argsort(descending=True)
    positions = [(idx.item() // 28, idx.item() % 28) for idx in ranked_indices]
    return positions, score


def try_n_pixels(n, ranked_positions, configs=None):
    """Try using top-n ranked pixels. Returns best accuracy."""
    positions = ranked_positions[:n]
    if configs is None:
        configs = [
            ([128, 64], 1e-3, 64, 0.0),
            ([256, 128], 1e-3, 64, 0.0),
            ([64, 32], 1e-3, 64, 0.0),
        ]

    best_acc = 0.0
    best_config = None
    for hidden, lr, bs, wd in configs:
        acc, params, elapsed = train_and_eval(
            positions, hidden, lr=lr, bs=bs, weight_decay=wd
        )
        print(f"  n={n} hidden={hidden} lr={lr} -> acc={acc:.4f} params={params} time={elapsed:.1f}s")
        if acc > best_acc:
            best_acc = acc
            best_config = (hidden, lr, bs, wd)
        if acc >= TARGET_ACC:
            break
    return best_acc, best_config, positions


def greedy_drop(positions, hidden_sizes, lr, bs, wd, target_acc=TARGET_ACC):
    """Greedily try to drop pixels one at a time."""
    current = list(positions)
    print(f"\n--- Greedy drop from {len(current)} pixels ---")

    while len(current) > 1:
        best_drop_acc = 0.0
        best_drop_idx = -1

        # Try dropping each pixel
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            acc, _, _ = train_and_eval(candidate, hidden_sizes, lr=lr, bs=bs,
                                        weight_decay=wd)
            if acc > best_drop_acc:
                best_drop_acc = acc
                best_drop_idx = i

        if best_drop_acc >= target_acc:
            dropped = current[best_drop_idx]
            current = current[:best_drop_idx] + current[best_drop_idx+1:]
            print(f"  Dropped pixel {dropped} -> {len(current)} pixels, acc={best_drop_acc:.4f}")
        else:
            print(f"  Cannot drop any pixel (best after drop: {best_drop_acc:.4f})")
            break

    return current


def greedy_swap(positions, all_ranked, hidden_sizes, lr, bs, wd):
    """Try swapping each position with unused high-ranked pixels."""
    current = list(positions)
    used = set(positions)
    # Candidates: next 50 ranked pixels not in current set
    candidates = [p for p in all_ranked if p not in used][:50]

    improved = True
    while improved:
        improved = False
        base_acc, _, _ = train_and_eval(current, hidden_sizes, lr=lr, bs=bs,
                                         weight_decay=wd)
        print(f"\n  Swap round: {len(current)} pixels, base_acc={base_acc:.4f}")

        for i in range(len(current)):
            best_swap_acc = base_acc
            best_swap_candidate = None

            for cand in candidates[:20]:  # limit to top 20 candidates
                trial = current[:i] + [cand] + current[i+1:]
                acc, _, _ = train_and_eval(trial, hidden_sizes, lr=lr, bs=bs,
                                            weight_decay=wd)
                if acc > best_swap_acc:
                    best_swap_acc = acc
                    best_swap_candidate = cand

            if best_swap_candidate is not None:
                old = current[i]
                current[i] = best_swap_candidate
                candidates = [c for c in candidates if c != best_swap_candidate]
                candidates.append(old)
                print(f"  Swapped {old} -> {best_swap_candidate}: acc={best_swap_acc:.4f}")
                improved = True
                break  # restart swap round

    return current


def run():
    print(f"=== Goal D1: Fewest input positions ===")
    print(f"Target: {TARGET_ACC*100:.0f}% accuracy")
    print()

    t_global = time.time()
    results = {}  # n -> (acc, config, positions)

    # Step 1: Rank pixels
    print("--- Step 1: Ranking pixels by importance ---")
    ranked_positions, scores = rank_pixels()
    print(f"Top 10 pixels: {ranked_positions[:10]}")
    print()

    # Step 2: Coarse binary search on number of pixels
    print("--- Step 2: Coarse binary search ---")
    lo, hi = 1, 200

    # Quick upper bound
    acc200, cfg200, pos200 = try_n_pixels(200, ranked_positions)
    results[200] = (acc200, cfg200, pos200)
    print(f"n=200: acc={acc200:.4f} {'PASS' if acc200 >= TARGET_ACC else 'FAIL'}")

    if acc200 < TARGET_ACC:
        acc400, cfg400, pos400 = try_n_pixels(400, ranked_positions)
        results[400] = (acc400, cfg400, pos400)
        hi = 400 if acc400 >= TARGET_ACC else 784

    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        elapsed = time.time() - t_global
        if elapsed > 12000:  # 3.3 hours safety margin
            print(f"Time limit approaching ({elapsed:.0f}s), stopping search")
            break

        print(f"\nTrying n={mid} (range [{lo}, {hi}])")
        acc, cfg, pos = try_n_pixels(mid, ranked_positions)
        results[mid] = (acc, cfg, pos)
        if acc >= TARGET_ACC:
            hi = mid
            print(f"  n={mid}: PASS ({acc:.4f}) -> search [{lo}, {hi}]")
        else:
            lo = mid + 1
            print(f"  n={mid}: FAIL ({acc:.4f}) -> search [{lo}, {hi}]")

    # Phase 3: Aggressive training at boundary
    print(f"\n--- Phase 3: Aggressive training at boundary n={lo} ---")
    boundary = lo
    aggressive_configs = [
        ([256, 128], 1e-3, 64, 0.0),
        ([512, 256], 1e-3, 64, 0.0),
        ([256, 128, 64], 1e-3, 64, 0.0),
        ([256, 128], 5e-4, 32, 0.0),
        ([512, 256, 128], 1e-3, 64, 0.0),
        ([256, 128], 1e-3, 64, 1e-5),
        ([256, 128], 2e-3, 128, 0.0),
    ]

    for offset in range(1, 8):
        n_try = boundary - offset
        if n_try < 1:
            break
        elapsed = time.time() - t_global
        if elapsed > 12600:
            print(f"Time limit ({elapsed:.0f}s), stopping")
            break

        print(f"\nAggressive try n={n_try}")
        acc, cfg, pos = try_n_pixels(n_try, ranked_positions, aggressive_configs)
        results[n_try] = (acc, cfg, pos)
        if acc >= TARGET_ACC:
            boundary = n_try
            print(f"  n={n_try}: PASS! New boundary = {boundary}")
        else:
            print(f"  n={n_try}: FAIL ({acc:.4f})")
            break

    # Phase 4: Greedy drop refinement
    elapsed = time.time() - t_global
    remaining = 14400 - elapsed  # 4 hour budget
    if remaining > 600 and boundary <= 100:
        print(f"\n--- Phase 4: Greedy drop from n={boundary} ({remaining:.0f}s remaining) ---")
        best_positions = ranked_positions[:boundary]
        # Find best config for this n
        passing = {n: r for n, r in results.items() if r[0] >= TARGET_ACC}
        if passing:
            min_n = min(passing.keys())
            _, best_cfg, _ = passing[min_n]
            if best_cfg:
                hidden, lr, bs, wd = best_cfg
                refined = greedy_drop(best_positions, hidden, lr, bs, wd)
                boundary = len(refined)
                results[f"drop_{boundary}"] = (TARGET_ACC, best_cfg, refined)
                ranked_positions_for_swap = ranked_positions  # keep for swap

    # Phase 5: Greedy swap to improve accuracy at current boundary
    elapsed = time.time() - t_global
    remaining = 14400 - elapsed
    if remaining > 600 and boundary <= 60:
        print(f"\n--- Phase 5: Greedy swap at n={boundary} ({remaining:.0f}s remaining) ---")
        if f"drop_{boundary}" in results:
            current_pos = results[f"drop_{boundary}"][2]
        else:
            current_pos = ranked_positions[:boundary]
        best_cfg_data = None
        for k, v in results.items():
            if v[0] >= TARGET_ACC and v[1] is not None:
                best_cfg_data = v[1]
                break
        if best_cfg_data:
            hidden, lr, bs, wd = best_cfg_data
            swapped = greedy_swap(current_pos, ranked_positions, hidden, lr, bs, wd)
            final_acc, _, _ = train_and_eval(swapped, hidden, lr=lr, bs=bs,
                                              weight_decay=wd)
            if final_acc >= TARGET_ACC:
                results[f"swap_{len(swapped)}"] = (final_acc, best_cfg_data, swapped)

    # Summary
    total_time = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.0f}s ({total_time/60:.1f} min)")

    passing = {}
    for k, v in results.items():
        if v[0] >= TARGET_ACC:
            if isinstance(k, int):
                passing[k] = v
            else:
                # drop_N or swap_N
                n = int(str(k).split('_')[1])
                passing[n] = v

    if passing:
        best_n = min(passing.keys())
        best_acc, best_cfg, best_pos = passing[best_n]
        print(f"FINAL RESULT: minimum positions = {best_n}")
        print(f"Accuracy: {best_acc:.4f}")
        print(f"Config: {best_cfg}")
        print(f"Positions ({best_n} pixels):")
        for i, p in enumerate(best_pos[:best_n]):
            print(f"  {p}")
    else:
        print("No configuration reached target accuracy!")

    print(f"\nAll results:")
    for k in sorted([k for k in results.keys() if isinstance(k, int)]):
        acc, cfg, _ = results[k]
        mark = "PASS" if acc >= TARGET_ACC else "FAIL"
        print(f"  n={k:4d}: acc={acc:.4f} [{mark}]")


if __name__ == "__main__":
    run()
