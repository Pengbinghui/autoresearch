#!/usr/bin/env python3
"""Batch experiment runner for Goal C: Fastest convergence to >= 98% accuracy."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def make_train_py_goalC(arch_code, lr=1e-3, bs=64, wd=0.0, eval_every=300, optimizer="Adam", schedule=None, extra_imports=""):
    sched_code = ""
    step_sched = ""
    if schedule == "onecycle":
        sched_code = f"scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr={lr*10}, total_steps=2000, pct_start=0.3)"
        step_sched = """
        scheduler.step()"""
    elif schedule == "cosine":
        sched_code = f"scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000)"
        step_sched = """
        scheduler.step()"""

    if optimizer == "Adam":
        opt_code = f"optimizer = torch.optim.Adam(model.parameters(), lr={lr}, weight_decay={wd})"
    elif optimizer == "AdamW":
        opt_code = f"optimizer = torch.optim.AdamW(model.parameters(), lr={lr}, weight_decay={wd})"
    elif optimizer == "SGD":
        opt_code = f"optimizer = torch.optim.SGD(model.parameters(), lr={lr}, momentum=0.9, weight_decay={wd})"
    else:
        opt_code = f"optimizer = torch.optim.Adam(model.parameters(), lr={lr}, weight_decay={wd})"

    return f'''"""MNIST search — training script."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy
{extra_imports}

TIME_BUDGET = 120
TARGET_ACC = 0.98

{arch_code}

t_start = time.time()
torch.manual_seed(42)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {{num_params:,}}")
train_images, train_labels, _, _ = get_data()
{opt_code}
{sched_code}

model.train()
epoch = 0
step = 0
t_train_start = time.time()
reached = False

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, {bs}):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step(){step_sched}
        step += 1
        training_time = time.time() - t_train_start
        if step % {eval_every} == 0:
            acc = evaluate_accuracy(model)
            model.train()
            print(f"step {{step}} | loss: {{loss.item():.4f}} | val_acc: {{acc:.4f}} | time: {{training_time:.1f}}s")
            if acc >= TARGET_ACC:
                reached = True
                break
        if training_time >= TIME_BUDGET:
            break
    if reached or training_time >= TIME_BUDGET:
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
        return "crash", 0.0, 0, 999.0, best_commit
    parts = out.split("|")
    status = parts[0]
    acc = float(parts[1])
    params = int(parts[2])
    secs = float(parts[3])
    commit = parts[4]
    return status, acc, params, secs, commit


def run_goalC():
    best_commit = "bb91efe"
    best_time = 999.0  # No valid baseline commit; find best from scratch
    print(f"=== GOAL C: Fastest convergence to >= 98% ===")
    print(f"Starting best: {best_time}s")

    experiments = []

    # =========================================================================
    # Architecture variations with current best hyperparams
    # =========================================================================

    # Current best: conv(8,16) fc64, lr=1e-3, bs=64, eval@300
    base_arch = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""

    # --- LR sweep on base arch ---
    for lr, label in [(2e-3, "lr=2e-3"), (3e-3, "lr=3e-3"), (5e-3, "lr=5e-3"),
                       (5e-4, "lr=5e-4"), (1.5e-3, "lr=1.5e-3"), (7e-4, "lr=7e-4")]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, lr, 64, 0, 300, "Adam", None))

    # --- Batch size sweep on base arch ---
    for bs, label in [(32, "bs=32"), (128, "bs=128"), (256, "bs=256"), (512, "bs=512")]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, 1e-3, bs, 0, 300, "Adam", None))

    # --- Eval frequency sweep ---
    for ev, label in [(100, "eval@100"), (200, "eval@200"), (400, "eval@400"), (500, "eval@500")]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, 1e-3, 64, 0, ev, "Adam", None))

    # --- Combined LR + BS ---
    for lr, bs, label in [(2e-3, 128, "lr=2e-3 bs=128"), (3e-3, 128, "lr=3e-3 bs=128"),
                           (2e-3, 256, "lr=2e-3 bs=256"), (3e-3, 256, "lr=3e-3 bs=256"),
                           (5e-3, 256, "lr=5e-3 bs=256"), (2e-3, 32, "lr=2e-3 bs=32"),
                           (1.5e-3, 128, "lr=1.5e-3 bs=128")]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, lr, bs, 0, 300, "Adam", None))

    # --- Optimizer variations ---
    experiments.append(("C: conv(8,16) fc64 AdamW", base_arch, 1e-3, 64, 1e-4, 300, "AdamW", None))
    experiments.append(("C: conv(8,16) fc64 AdamW lr=2e-3", base_arch, 2e-3, 64, 1e-4, 300, "AdamW", None))

    # --- LR schedule ---
    experiments.append(("C: conv(8,16) fc64 onecycle", base_arch, 1e-3, 64, 0, 300, "Adam", "onecycle"))
    experiments.append(("C: conv(8,16) fc64 cosine", base_arch, 1e-3, 64, 0, 300, "Adam", "cosine"))

    # =========================================================================
    # Architecture variations
    # =========================================================================

    # Wider first layer
    for c1, c2, fc, label in [
        (10, 20, 64, "conv(10,20) fc64"),
        (12, 16, 64, "conv(12,16) fc64"),
        (6, 16, 64, "conv(6,16) fc64"),
        (8, 16, 32, "conv(8,16) fc32"),
        (8, 16, 128, "conv(8,16) fc128"),
        (8, 24, 64, "conv(8,24) fc64"),
        (16, 16, 64, "conv(16,16) fc64"),
        (8, 32, 64, "conv(8,32) fc64"),
        (6, 12, 48, "conv(6,12) fc48"),
    ]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * 7 * 7, {fc})
        self.fc2 = nn.Linear({fc}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append(("C: " + label, arch, 1e-3, 64, 0, 300, "Adam", None))

    # Architecture + best hyperparams combos
    for c1, c2, fc, lr, bs, label in [
        (10, 20, 64, 2e-3, 128, "conv(10,20) fc64 lr=2e-3 bs=128"),
        (8, 24, 64, 2e-3, 64, "conv(8,24) fc64 lr=2e-3"),
        (8, 32, 64, 2e-3, 64, "conv(8,32) fc64 lr=2e-3"),
        (6, 16, 64, 2e-3, 64, "conv(6,16) fc64 lr=2e-3"),
        (8, 16, 128, 2e-3, 64, "conv(8,16) fc128 lr=2e-3"),
        (12, 16, 64, 2e-3, 128, "conv(12,16) fc64 lr=2e-3 bs=128"),
    ]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * 7 * 7, {fc})
        self.fc2 = nn.Linear({fc}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append(("C: " + label, arch, lr, bs, 0, 300, "Adam", None))

    # BatchNorm variants (may help convergence speed)
    for c1, c2, fc, label in [
        (8, 16, 64, "conv(8,16)+BN fc64"),
        (10, 20, 64, "conv(10,20)+BN fc64"),
    ]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.bn1 = nn.BatchNorm2d({c1})
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.bn2 = nn.BatchNorm2d({c2})
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * 7 * 7, {fc})
        self.fc2 = nn.Linear({fc}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append(("C: " + label, arch, 1e-3, 64, 0, 300, "Adam", None))

    # BN + higher LR (BN enables higher LR)
    for c1, c2, fc, lr, label in [
        (8, 16, 64, 3e-3, "conv(8,16)+BN fc64 lr=3e-3"),
        (8, 16, 64, 5e-3, "conv(8,16)+BN fc64 lr=5e-3"),
        (10, 20, 64, 3e-3, "conv(10,20)+BN fc64 lr=3e-3"),
    ]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 3, padding=1)
        self.bn1 = nn.BatchNorm2d({c1})
        self.conv2 = nn.Conv2d({c1}, {c2}, 3, padding=1)
        self.bn2 = nn.BatchNorm2d({c2})
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * 7 * 7, {fc})
        self.fc2 = nn.Linear({fc}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append(("C: " + label, arch, lr, 64, 0, 300, "Adam", None))

    # 5x5 kernel (fewer layers, larger receptive field)
    for c1, c2, fc, label in [
        (8, 16, 64, "conv5x5(8,16) fc64"),
        (10, 20, 64, "conv5x5(10,20) fc64"),
    ]:
        arch = f"""class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, {c1}, 5, padding=2)
        self.conv2 = nn.Conv2d({c1}, {c2}, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear({c2} * 7 * 7, {fc})
        self.fc2 = nn.Linear({fc}, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
        experiments.append(("C: " + label, arch, 1e-3, 64, 0, 300, "Adam", None))

    # Larger batch + higher LR (use all good combos from hyperparameter sweeps)
    for lr, bs, ev, label in [
        (3e-3, 128, 200, "lr=3e-3 bs=128 eval@200"),
        (5e-3, 256, 200, "lr=5e-3 bs=256 eval@200"),
        (2e-3, 128, 200, "lr=2e-3 bs=128 eval@200"),
        (3e-3, 64, 200, "lr=3e-3 eval@200"),
        (2e-3, 64, 200, "lr=2e-3 eval@200"),
        (1e-3, 64, 200, "eval@200"),
    ]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, lr, bs, 0, ev, "Adam", None))

    # Weight decay variations
    for wd, label in [(1e-4, "wd=1e-4"), (1e-5, "wd=1e-5")]:
        experiments.append(("C: conv(8,16) fc64 " + label, base_arch, 1e-3, 64, wd, 300, "Adam", None))

    # Mixed kernel: 5x5 first, 3x3 second
    arch_mixed = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
    experiments.append(("C: conv5x3(8,16) fc64", arch_mixed, 1e-3, 64, 0, 300, "Adam", None))
    experiments.append(("C: conv5x3(8,16) fc64 lr=2e-3", arch_mixed, 2e-3, 64, 0, 300, "Adam", None))

    # 3-conv architecture (may converge faster with deeper features)
    arch_3conv = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
    experiments.append(("C: 3conv(8,16,32) fc32", arch_3conv, 1e-3, 64, 0, 300, "Adam", None))
    experiments.append(("C: 3conv(8,16,32) fc32 lr=2e-3", arch_3conv, 2e-3, 64, 0, 300, "Adam", None))

    # Best arch with eval@150 (fine-grained early stop)
    experiments.append(("C: conv(8,16) fc64 eval@150", base_arch, 1e-3, 64, 0, 150, "Adam", None))
    experiments.append(("C: conv(8,16) fc64 lr=2e-3 eval@150", base_arch, 2e-3, 64, 0, 150, "Adam", None))

    # Wider single FC layer (no fc2, just one big linear)
    arch_1fc = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 7 * 7, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x"""
    experiments.append(("C: conv(8,16) fc-direct", arch_1fc, 1e-3, 64, 0, 300, "Adam", None))

    count = 0
    for exp in experiments:
        desc, arch, lr, bs, wd, eval_every, opt, sched = exp
        code = make_train_py_goalC(arch, lr=lr, bs=bs, wd=wd, eval_every=eval_every, optimizer=opt, schedule=sched)
        status, acc, params, secs, commit = run_one(best_commit, desc, code)
        count += 1
        marker = ""

        if status == "keep" and secs < best_time:
            best_time = secs
            best_commit = commit
            marker = " *** NEW BEST ***"
        elif status == "keep":
            subprocess.run(["git", "reset", "--hard", best_commit], capture_output=True, cwd=os.getcwd())

        print(f"[{count}] {desc}: {status} acc={acc:.4f} time={secs}s params={params}{marker}")

    print(f"\n=== GOAL C: {count} experiments ===")
    print(f"Best: {best_time}s (commit {best_commit})")


if __name__ == "__main__":
    run_goalC()
