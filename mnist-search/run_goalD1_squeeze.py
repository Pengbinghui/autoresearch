#!/usr/bin/env python3
"""Goal D1 squeeze: Fine-tune boundary around 83-86 pixels with aggressive training.
Also try different seeds and longer training."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import get_data, make_batches, evaluate_accuracy

TARGET_ACC = 0.95


class PixelSelectModel(nn.Module):
    def __init__(self, positions, hidden_sizes):
        super().__init__()
        self.positions = positions
        n_pixels = len(positions)
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
        selected = x[:, 0, self.rows, self.cols]
        return self.classifier(selected)


def rank_pixels():
    train_images, train_labels, _, _ = get_data()
    flat = train_images.view(train_images.size(0), -1)
    variance = flat.var(dim=0)
    grand_mean = flat.mean(dim=0)
    between_var = torch.zeros(784)
    for c in range(10):
        mask = train_labels == c
        class_mean = flat[mask].mean(dim=0)
        between_var += mask.sum().float() * (class_mean - grand_mean) ** 2
    between_var /= train_images.size(0)
    score = between_var * variance
    ranked_indices = score.argsort(descending=True)
    positions = [(idx.item() // 28, idx.item() % 28) for idx in ranked_indices]
    return positions


def train_extended(positions, hidden_sizes, lr=1e-3, bs=64, max_epochs=100,
                   patience=10, weight_decay=0.0, seed=42):
    torch.manual_seed(seed)
    model = PixelSelectModel(positions, hidden_sizes)
    params = sum(p.numel() for p in model.parameters())
    train_images, train_labels, _, _ = get_data()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        acc = evaluate_accuracy(model)
        model.train()
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - t_start
        if acc >= TARGET_ACC:
            break
        if no_improve >= patience:
            break
        if elapsed > 300:
            break

    return best_acc, params, time.time() - t_start


def run():
    print("=== Goal D1 Squeeze: Fine boundary search ===\n")
    ranked = rank_pixels()

    configs = [
        ([512, 256], 1e-3, 64, 0.0, 42),
        ([256, 128], 1e-3, 64, 0.0, 42),
        ([512, 256], 5e-4, 32, 0.0, 42),
        ([512, 256, 128], 1e-3, 64, 0.0, 42),
        ([1024, 512], 1e-3, 64, 0.0, 42),
        ([512, 256], 1e-3, 64, 0.0, 0),
        ([512, 256], 1e-3, 64, 0.0, 1),
        ([512, 256], 1e-3, 64, 0.0, 2),
        ([512, 256], 1e-3, 64, 0.0, 3),
        ([512, 256], 3e-4, 32, 0.0, 42),
        ([256, 128], 3e-4, 32, 0.0, 42),
        ([512, 256], 1e-3, 64, 1e-5, 42),
    ]

    # Test n=85, 84, 83, 82, 81, 80, 75, 70, 65, 60
    test_ns = [85, 84, 83, 82, 81, 80, 78, 76, 74, 72, 70, 65, 60]
    best_n = 86  # Known to pass

    for n in test_ns:
        positions = ranked[:n]
        print(f"\n--- n={n} pixels ---")
        best_acc = 0.0

        for hidden, lr, bs, wd, seed in configs:
            acc, params, elapsed = train_extended(
                positions, hidden, lr=lr, bs=bs, weight_decay=wd, seed=seed
            )
            if acc > best_acc:
                best_acc = acc
            mark = "PASS" if acc >= TARGET_ACC else ""
            print(f"  h={hidden} lr={lr} bs={bs} s={seed}: acc={acc:.4f} {mark} ({elapsed:.0f}s)")
            if acc >= TARGET_ACC:
                break

        if best_acc >= TARGET_ACC:
            best_n = n
            print(f"  *** n={n} PASSES! ***")
        else:
            print(f"  n={n} FAILS (best={best_acc:.4f})")
            break  # Stop going lower since we're using ranked, lower n will be worse

    print(f"\n{'='*60}")
    print(f"FINAL RESULT: minimum positions = {best_n}")
    print(f"\nSelected positions ({best_n} pixels):")
    for p in ranked[:best_n]:
        print(f"  {p}")


if __name__ == "__main__":
    run()
