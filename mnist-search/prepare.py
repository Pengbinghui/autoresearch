"""
MNIST search — data prep and evaluation.
One-time setup: downloads MNIST dataset.
Runtime: provides pre-loaded data tensors and the fixed evaluation metric.
Usage: uv run prepare.py
"""

import os
import time

import torch
from torchvision import datasets

# ---------------------------------------------------------------------------
# Fixed constants (do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 60  # training time budget in seconds (wall clock)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Data — pre-loaded into memory for fast access
# ---------------------------------------------------------------------------

_cache = {}

def get_data():
    """
    Returns (train_images, train_labels, val_images, val_labels).
    All tensors, pre-normalized, loaded into memory once.
    Images: float32, shape [N, 1, 28, 28], normalized (mean=0.1307, std=0.3081).
    Labels: long, shape [N].
    """
    if "train_images" not in _cache:
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True)
        val_ds = datasets.MNIST(DATA_DIR, train=False, download=True)
        # Normalize: [0,255] uint8 -> float32, then standard MNIST normalization
        _cache["train_images"] = train_ds.data.unsqueeze(1).float().div(255).sub(0.1307).div(0.3081)
        _cache["train_labels"] = train_ds.targets
        _cache["val_images"] = val_ds.data.unsqueeze(1).float().div(255).sub(0.1307).div(0.3081)
        _cache["val_labels"] = val_ds.targets
    return _cache["train_images"], _cache["train_labels"], _cache["val_images"], _cache["val_labels"]

def make_batches(images, labels, batch_size, shuffle=True):
    """Yield (x, y) batches from pre-loaded tensors. One pass through the data."""
    n = len(images)
    if shuffle:
        idx = torch.randperm(n)
    else:
        idx = torch.arange(n)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield images[batch_idx], labels[batch_idx]

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model, batch_size=1000):
    """
    Accuracy on the full MNIST test set (10,000 images).
    This is the authoritative metric — higher is better.
    """
    _, _, val_images, val_labels = get_data()
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for x, y in make_batches(val_images, val_labels, batch_size, shuffle=False):
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    if was_training:
        model.train()
    return correct / total

# ---------------------------------------------------------------------------
# Main — verify data is downloaded
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Downloading MNIST dataset...")
    t0 = time.time()
    train_img, train_lbl, val_img, val_lbl = get_data()
    print(f"Train: {len(train_img):,} images, Val: {len(val_img):,} images")
    print(f"Image shape: {train_img.shape}, dtype: {train_img.dtype}")
    print(f"Done in {time.time() - t0:.1f}s")
    print(f"Data stored at: {DATA_DIR}")
