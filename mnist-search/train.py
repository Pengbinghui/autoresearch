"""
MNIST search — training script. Single file, CPU only.
The agent modifies this file to improve val_accuracy.
Usage: uv run train.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

# ---------------------------------------------------------------------------
# Model (edit this — architecture is fair game)
# ---------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
OPTIMIZER = "adam"      # "adam" or "sgd"
SGD_MOMENTUM = 0.9      # only used if OPTIMIZER == "sgd"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

train_images, train_labels, _, _ = get_data()

if OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM, weight_decay=WEIGHT_DECAY)

# ---------------------------------------------------------------------------
# Training loop (wall-clock time budget)
# ---------------------------------------------------------------------------

model.train()
epoch = 0
step = 0
t_train_start = time.time()

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, BATCH_SIZE):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        step += 1

        training_time = time.time() - t_train_start
        if training_time >= TIME_BUDGET:
            break

    progress = min(training_time / TIME_BUDGET, 1.0)
    print(f"\repoch {epoch} | step {step} | loss: {loss.item():.4f} | {100*progress:.0f}%    ", end="", flush=True)

    if training_time >= TIME_BUDGET:
        break

print()
training_time = time.time() - t_train_start

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

val_accuracy = evaluate_accuracy(model)
t_end = time.time()

print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_epochs:     {epoch}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
