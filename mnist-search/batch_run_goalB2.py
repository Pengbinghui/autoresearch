#!/usr/bin/env python3
"""Supplementary Goal B experiments: boundary search around best (1368 params)."""
import subprocess, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import shared helpers
from batch_run import make_train_py_goalB, run_one

def run_goalB2():
    best_commit = "6fb353f"  # 3conv(3,6,8) 3pool = 1368 params
    best_params = 1368
    print(f"=== GOAL B ROUND 2: Boundary search ===")
    print(f"Starting best: {best_params} params")

    experiments = []

    # Try architectures between 1122 (failed) and 1368 (success)
    for c1, c2, c3 in [
        (3, 6, 7),  # 1223 params
        (3, 5, 8),  # 1268 params
        (2, 6, 8),  # 1304 params
        (2, 5, 8),  # 1213 params
        (3, 4, 8),  # 1168 params
        (3, 5, 7),  # 1132 params
        (3, 6, 6),  # 1078 params
        (3, 5, 6),  # 996 params
        (4, 6, 8),  # 1432 params (slightly bigger, may be more reliable)
    ]:
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
        experiments.append((f"B2: 3conv({c1},{c2},{c3}) 3pool fc({c3}*9,10)", arch))

    # Try the best arch with better hyperparams
    best_arch = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 3 * 3, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x"""
    for lr, bs, wd, pat, label in [
        (5e-4, 64, 0, 5, "lr=5e-4 pat=5"),
        (1e-3, 32, 0, 5, "bs=32 pat=5"),
        (1e-3, 64, 1e-4, 5, "wd=1e-4 pat=5"),
        (1e-3, 64, 0, 8, "pat=8"),
    ]:
        experiments.append((f"B2: 3conv(3,6,8) 3pool {label}", best_arch, lr, bs, wd, pat))

    # (2,4,8) was close at 97.57% — try with better hyperparams
    near_arch = """class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 3 * 3, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x"""
    for lr, bs, wd, pat, label in [
        (5e-4, 64, 0, 5, "lr=5e-4 pat=5"),
        (1e-3, 32, 0, 5, "bs=32 pat=5"),
        (1e-3, 64, 1e-4, 5, "wd=1e-4 pat=5"),
    ]:
        experiments.append((f"B2: 3conv(2,4,8) 3pool {label}", near_arch, lr, bs, wd, pat))

    count = 0
    for exp in experiments:
        if len(exp) == 2:
            desc, arch = exp
            lr, bs, wd, pat = 1e-3, 64, 0.0, 3
        elif len(exp) == 6:
            desc, arch, lr, bs, wd, pat = exp
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

    print(f"\n=== GOAL B ROUND 2: {count} experiments ===")
    print(f"Best: {best_params} params (commit {best_commit})")


if __name__ == "__main__":
    run_goalB2()
