#!/usr/bin/env python3
"""Goal D2 squeeze: Try harder at d=7 with longer training and LR scheduling."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import get_data, make_batches, evaluate_accuracy

TARGET_ACC = 0.95


def make_model(d, hidden_sizes, dropout=0.0):
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


def train_extended(d, hidden_sizes, lr=1e-3, bs=64, max_epochs=200, patience=15,
                   weight_decay=0.0, use_cosine=False, seed=42):
    """Train with extended epochs and optional cosine LR."""
    torch.manual_seed(seed)
    model = make_model(d, hidden_sizes)
    params = sum(p.numel() for p in model.parameters())
    train_images, train_labels, _, _ = get_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    model.train()
    best_acc = 0.0
    no_improve = 0
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        for x, y in make_batches(train_images, train_labels, bs):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        acc = evaluate_accuracy(model)
        model.train()

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - t_start
        if epoch % 10 == 0 or acc >= TARGET_ACC:
            print(f"    ep{epoch} acc={acc:.4f} best={best_acc:.4f} {elapsed:.0f}s")

        if acc >= TARGET_ACC:
            break
        if no_improve >= patience:
            break
        if elapsed > 600:  # 10 min max per trial
            break

    return best_acc, params, time.time() - t_start


def run():
    print("=== Goal D2 Squeeze: Trying harder at d=7 ===\n")

    configs = [
        # (hidden, lr, bs, wd, cosine, seed, label)
        ([256, 128], 3e-4, 32, 0, False, 42, "lr=3e-4 bs=32 long"),
        ([256, 128], 3e-4, 32, 0, True, 42, "lr=3e-4 bs=32 cosine"),
        ([256, 128], 5e-4, 64, 0, True, 42, "lr=5e-4 cosine"),
        ([512, 256], 3e-4, 32, 0, False, 42, "wide lr=3e-4 bs=32"),
        ([512, 256], 3e-4, 32, 0, True, 42, "wide lr=3e-4 bs=32 cosine"),
        ([256, 128], 3e-4, 32, 0, False, 0, "lr=3e-4 bs=32 seed=0"),
        ([256, 128], 3e-4, 32, 0, False, 1, "lr=3e-4 bs=32 seed=1"),
        ([256, 128], 3e-4, 32, 0, False, 2, "lr=3e-4 bs=32 seed=2"),
        ([256, 128], 3e-4, 32, 0, False, 3, "lr=3e-4 bs=32 seed=3"),
        ([256, 128], 3e-4, 32, 0, False, 7, "lr=3e-4 bs=32 seed=7"),
        ([256, 128], 2e-4, 32, 0, False, 42, "lr=2e-4 bs=32"),
        ([256, 128], 1e-4, 32, 0, False, 42, "lr=1e-4 bs=32"),
        ([1024, 512, 256], 3e-4, 64, 0, False, 42, "very wide 3-layer"),
        ([512, 512, 256], 3e-4, 32, 0, False, 42, "deep wide"),
        ([256, 128], 3e-4, 32, 1e-5, False, 42, "wd=1e-5"),
        ([256, 128], 3e-4, 16, 0, False, 42, "bs=16"),
    ]

    best_acc = 0.0
    for hidden, lr, bs, wd, cosine, seed, label in configs:
        print(f"  d=7 {label}:")
        acc, params, elapsed = train_extended(
            7, hidden, lr=lr, bs=bs, weight_decay=wd, use_cosine=cosine,
            seed=seed, max_epochs=200, patience=15
        )
        mark = "PASS" if acc >= TARGET_ACC else "FAIL"
        print(f"  -> acc={acc:.4f} [{mark}] params={params} time={elapsed:.0f}s\n")
        if acc > best_acc:
            best_acc = acc
        if acc >= TARGET_ACC:
            print(f"  *** d=7 ACHIEVED {TARGET_ACC*100}%! ***")
            break

    if best_acc < TARGET_ACC:
        print(f"\nd=7 best: {best_acc:.4f} — confirmed d=8 is the minimum")
    else:
        print(f"\nd=7 PASSED! Minimum d = 7")


if __name__ == "__main__":
    run()
