#!/usr/bin/env python3
"""Batch experiment runner for Goal B and Goal C."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def make_train_py_goalB(arch_code, lr=1e-3, bs=64, wd=0.0, patience=3):
    return f'''"""MNIST search — training script."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 120

{arch_code}

t_start = time.time()
torch.manual_seed(42)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {{num_params:,}}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr={lr}, weight_decay={wd})

model.train()
epoch = 0
step = 0
best_acc = 0.0
patience = {patience}
no_improve = 0
t_train_start = time.time()

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, {bs}):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        step += 1
        training_time = time.time() - t_train_start
        if training_time >= TIME_BUDGET:
            break
    progress = min(training_time / TIME_BUDGET, 1.0)
    print(f"\\repoch {{epoch}} | step {{step}} | loss: {{loss.item():.4f}} | {{100*progress:.0f}}%    ", end="", flush=True)
    acc = evaluate_accuracy(model)
    model.train()
    print(f" | val_acc: {{acc:.4f}}")
    if acc > best_acc:
        best_acc = acc
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= patience:
        break
    if training_time >= TIME_BUDGET:
        break

print()
training_time = time.time() - t_train_start
val_accuracy = evaluate_accuracy(model)
t_end = time.time()
print("---")
print(f"val_accuracy:     {{val_accuracy:.6f}}")
print(f"training_seconds: {{training_time:.1f}}")
print(f"total_seconds:    {{t_end - t_start:.1f}}")
print(f"total_epochs:     {{epoch}}")
print(f"num_steps:        {{step}}")
print(f"num_params:       {{num_params}}")
'''

def run_one(best_commit, desc, code):
    """Run a single experiment. Returns (status, acc, params, time, commit)."""
    proc = subprocess.run(
        ["bash", "run_batch.sh", best_commit, desc],
        input=code, capture_output=True, text=True, timeout=300
    )
    out = proc.stdout.strip()
    if not out or "|" not in out:
        return "crash", 0.0, 0, 0.0, best_commit
    parts = out.split("|")
    status = parts[0]
    acc = float(parts[1])
    params = int(parts[2])
    secs = float(parts[3])
    commit = parts[4]
    return status, acc, params, secs, commit


def run_goalB():
    best_commit = "1b38828"
    best_params = 5088
    print(f"=== GOAL B: Smallest model >= 98% ===")
    print(f"Starting best: {best_params} params")

    # Define architectures to test
    experiments = []

    # 3-conv with 3 pools (28->14->7->3)
    for c1, c2, c3 in [(2,4,8), (2,4,16), (3,6,12), (3,6,16), (3,6,8), (2,6,12), (2,4,12), (3,8,16), (4,8,16), (2,6,16)]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.conv3 = nn.Conv2d({c2}, {c3}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear({c3} * 3 * 3, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x"""
        experiments.append((f"B: 3conv({c1},{c2},{c3}) 3pool fc({c3}*9,10)", arch))

    # 3-conv with 3 pools + hidden FC
    for c1, c2, c3, h in [(2,4,8,16), (3,6,8,16), (2,4,16,16), (3,6,12,16), (2,6,12,16)]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.conv3 = nn.Conv2d({c2}, {c3}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c3} * 3 * 3, {h})
        self.fc2 = nn.Linear({h}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append((f"B: 3conv({c1},{c2},{c3}) 3pool fc({c3}*9,{h},10)", arch))

    # 2-conv with adaptpool
    for c1, c2, p in [(3,6,3), (3,6,4), (3,6,5), (3,8,3), (3,8,4), (4,8,3), (4,8,4), (4,10,3), (4,10,4), (3,10,3)]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear({c2} * {p} * {p}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, {p})
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x"""
        experiments.append((f"B: conv({c1},{c2}) adaptpool({p}) fc({c2}*{p}*{p},10)", arch))

    # 2-conv with adaptpool + hidden FC
    for c1, c2, p, h in [(3,6,3,16), (3,6,4,16), (3,8,3,16), (4,8,3,16), (3,6,3,12), (3,8,4,12), (3,6,2,16)]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * {p} * {p}, {h})
        self.fc2 = nn.Linear({h}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, {p})
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append((f"B: conv({c1},{c2}) adaptpool({p}) fc({c2}*{p}*{p},{h},10)", arch))

    # Hyperparameter variations on current best (conv(3,6) fc16)
    for lr, bs, wd, pat, label in [
        (2e-3, 64, 0, 3, "lr=2e-3"),
        (5e-4, 64, 0, 3, "lr=5e-4"),
        (1e-3, 32, 0, 3, "bs=32"),
        (1e-3, 128, 0, 3, "bs=128"),
        (1e-3, 64, 1e-4, 3, "wd=1e-4"),
        (1e-3, 64, 0, 5, "patience=5"),
        (1e-3, 64, 0, 8, "patience=8"),
        (5e-4, 32, 0, 5, "lr=5e-4 bs=32 pat=5"),
        (2e-3, 32, 0, 3, "lr=2e-3 bs=32"),
    ]:
        experiments.append((f"B: conv(3,6) fc16 {label}", None, lr, bs, wd, pat))

    # conv(3,5) boundary with better hparams
    for lr, bs, wd, pat, label in [
        (5e-4, 64, 0, 5, "lr=5e-4 pat=5"),
        (5e-4, 32, 0, 5, "lr=5e-4 bs=32 pat=5"),
        (1e-3, 32, 0, 5, "bs=32 pat=5"),
        (1e-3, 64, 1e-5, 5, "wd=1e-5 pat=5"),
        (7e-4, 64, 0, 5, "lr=7e-4 pat=5"),
        (1e-3, 48, 0, 5, "bs=48 pat=5"),
    ]:
        arch = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 5, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append((f"B: conv(3,5) fc16 {label}", arch, lr, bs, wd, pat))

    count = 0
    for exp in experiments:
        if len(exp) == 2:
            desc, arch = exp
            lr, bs, wd, pat = 1e-3, 64, 0.0, 3
        elif len(exp) == 6:
            desc, arch_or_none, lr, bs, wd, pat = exp
            if arch_or_none is None:
                arch = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 7 * 7, 16)
        self.fc2 = nn.Linear(16, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
            else:
                arch = arch_or_none
        else:
            continue

        code = make_train_py_goalB(arch, lr=lr, bs=bs, wd=wd, patience=pat)
        status, acc, params, secs, commit = run_one(best_commit, desc, code)
        count += 1
        marker = ""

        if status == "keep" and params < best_params:
            best_params = params
            best_commit = commit
            marker = " *** NEW BEST ***"
        elif status == "keep":
            subprocess.run(["git", "reset", "--hard", best_commit], capture_output=True, cwd=os.getcwd())

        print(f"[{count}] {desc}: {status} acc={acc:.4f} params={params} time={secs}s{marker}")

    print(f"\n=== GOAL B: {count} experiments ===")
    print(f"Best: {best_params} params (commit {best_commit})")


if __name__ == "__main__":
    run_goalB()
