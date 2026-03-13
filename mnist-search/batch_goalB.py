"""
Batch experiment runner for Goal B (smallest model >= 98% accuracy).
Generates train.py variants, commits, runs, logs results.
"""
import subprocess, os, time, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BEST_COMMIT = "1b38828"
RESULTS_FILE = "results.tsv"
TRAIN_TEMPLATE = '''"""
MNIST search — training script. Single file, CPU only.
Usage: uv run train.py
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 120

class Net(nn.Module):
    def __init__(self):
        super().__init__()
{layers}

    def forward(self, x):
{forward}

BATCH_SIZE = {batch_size}
LEARNING_RATE = {lr}
WEIGHT_DECAY = {wd}

t_start = time.time()
torch.manual_seed(42)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {{num_params:,}}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

model.train()
epoch = 0
step = 0
best_acc = 0.0
patience = {patience}
no_improve = 0
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
        print(f"Early stopping: no improvement for {{patience}} epochs")
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

def run_experiment(desc, layers, forward, lr=1e-3, batch_size=64, wd=0.0, patience=3, best_commit=None):
    """Run a single experiment. Returns (commit, acc, params, time, status)."""
    if best_commit is None:
        best_commit = BEST_COMMIT

    code = TRAIN_TEMPLATE.format(
        layers=layers, forward=forward,
        lr=lr, batch_size=batch_size, wd=wd, patience=patience
    )
    with open("train.py", "w") as f:
        f.write(code)

    # Commit
    subprocess.run(["git", "add", "train.py"], capture_output=True)
    subprocess.run(["git", "commit", "-m", desc], capture_output=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True).stdout.strip()

    # Run with timeout
    try:
        result = subprocess.run(["uv", "run", "train.py"],
                               capture_output=True, text=True, timeout=200)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = ""

    # Extract metrics
    acc = "0.000000"
    params = "0"
    secs = "0.0"
    for line in output.split("\n"):
        if line.startswith("val_accuracy:"):
            acc = line.split()[1]
        elif line.startswith("num_params:"):
            params = line.split()[1]
        elif line.startswith("training_seconds:"):
            secs = line.split()[1]

    acc_f = float(acc)
    params_i = int(params)
    status = "keep" if acc_f >= 0.98 else ("crash" if acc_f == 0 else "discard")

    # Log
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{acc}\t{params}\t{secs}\t{status}\t{desc}\n")

    print(f"  {desc}: acc={acc} params={params} time={secs}s [{status}]")

    # Reset if not keeping
    if status != "keep":
        subprocess.run(["git", "reset", "--hard", best_commit], capture_output=True)

    return commit, acc_f, params_i, float(secs), status


if __name__ == "__main__":
    best_params = 5088
    best_acc = 0.9805
    best_commit_current = BEST_COMMIT
    exp_num = 2  # continuing from B02

    experiments = [
        # Group 1: 3rd pooling to reduce spatial dims
        ("B03: conv(3,6) pool3 fc(6*3*3,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.pool3 = nn.MaxPool2d(3, 3)\n        self.fc1 = nn.Linear(6 * 2 * 2, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool3(x)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 2: adaptive avg pool to reduce spatial
        ("B04: conv(3,6) adaptpool(2) fc(24,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 2 * 2, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 2)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        ("B05: conv(3,6) adaptpool(3) fc(54,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 3 * 3, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 3)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        ("B06: conv(3,6) adaptpool(4) fc(96,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 4 * 4, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 4)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 3: Direct output (no hidden FC)
        ("B07: conv(3,6) fc(294,10) no hidden layer",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(6 * 7 * 7, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B08: conv(4,8) adaptpool(2) fc(32,10) no hidden",
         "        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)\n        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(8 * 2 * 2, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 2)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        # Group 4: Stride-2 conv (no separate pool layer)
        ("B09: stride2conv(3,6) fc(6*7*7,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, stride=2, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, stride=2, padding=1)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = F.relu(self.conv1(x))\n        x = F.relu(self.conv2(x))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 5: Pool more aggressively (pool 4x4)
        ("B10: conv(3,6) pool4 fc(6*1*1,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(4, 4)\n        self.fc1 = nn.Linear(6 * 1 * 1, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 6: More channels with adaptpool reduction
        ("B11: conv(4,10) adaptpool(2) fc(40,10)",
         "        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)\n        self.conv2 = nn.Conv2d(4, 10, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(10 * 2 * 2, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 2)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B12: conv(4,10) adaptpool(3) fc(90,10)",
         "        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)\n        self.conv2 = nn.Conv2d(4, 10, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(10 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 3)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        # Group 7: Higher LR on current best
        ("B13: conv(3,6) fc16 lr=2e-3",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"lr": 2e-3}),

        ("B14: conv(3,6) fc16 lr=5e-4",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"lr": 5e-4}),

        # Group 8: Patience variations
        ("B15: conv(3,6) fc16 patience=5",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"patience": 5}),

        # Group 9: Try smaller fc1 with more channels
        ("B16: conv(4,8) adaptpool(3) fc(72,16,10)",
         "        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)\n        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(8 * 3 * 3, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 3)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        ("B17: conv(4,8) adaptpool(2) fc(32,16,10)",
         "        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)\n        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(8 * 2 * 2, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 2)\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 10: 3-conv architectures with pooling
        ("B18: 3conv(3,6,12) pool-pool-pool fc(12,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.conv3 = nn.Conv2d(6, 12, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(12 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B19: 3conv(3,6,12) pool-pool-pool fc(108,16,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.conv3 = nn.Conv2d(6, 12, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(12 * 3 * 3, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        ("B20: 3conv(2,4,8) pool-pool-pool fc(72,10)",
         "        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)\n        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)\n        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(8 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B21: 3conv(2,4,16) pool-pool-pool fc(144,10)",
         "        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)\n        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)\n        self.conv3 = nn.Conv2d(4, 16, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(16 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B22: 3conv(3,6,16) pool-pool-pool fc(144,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.conv3 = nn.Conv2d(6, 16, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(16 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = self.pool(F.relu(self.conv3(x)))\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        # Group 11: Asymmetric channels
        ("B23: conv(2,8) fc(8*7*7,16,10)",
         "        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)\n        self.conv2 = nn.Conv2d(2, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(8 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        ("B24: conv(2,6) fc(6*7*7,16,10)",
         "        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)\n        self.conv2 = nn.Conv2d(2, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)"),

        # Group 12: adaptpool on smaller convs
        ("B25: conv(3,8) adaptpool(3) fc(72,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(8 * 3 * 3, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 3)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        ("B26: conv(3,8) adaptpool(4) fc(128,10)",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 8, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc = nn.Linear(8 * 4 * 4, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = F.adaptive_avg_pool2d(x, 4)\n        x = x.view(x.size(0), -1)\n        x = self.fc(x)"),

        # Group 13: Weight decay variations
        ("B27: conv(3,6) fc16 wd=1e-4",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"wd": 1e-4}),

        # Group 14: Batch size variations
        ("B28: conv(3,6) fc16 bs=32",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"batch_size": 32}),

        ("B29: conv(3,6) fc16 bs=128",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(6 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"batch_size": 128}),

        # Group 15: conv(3,5) boundary experiments with different hyperparameters
        ("B30: conv(3,5) fc16 lr=5e-4",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 5, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(5 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"lr": 5e-4}),

        ("B31: conv(3,5) fc16 bs=32 lr=5e-4",
         "        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)\n        self.conv2 = nn.Conv2d(3, 5, 3, padding=1)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.fc1 = nn.Linear(5 * 7 * 7, 16)\n        self.fc2 = nn.Linear(16, 10)",
         "        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(x.size(0), -1)\n        x = F.relu(self.fc1(x))\n        x = self.fc2(x)",
         {"lr": 5e-4, "batch_size": 32}),
    ]

    for exp in experiments:
        if len(exp) == 3:
            desc, layers, forward = exp
            kwargs = {}
        else:
            desc, layers, forward, kwargs = exp

        commit, acc, params, secs, status = run_experiment(
            desc, layers, forward,
            best_commit=best_commit_current,
            **kwargs
        )

        if status == "keep" and params < best_params:
            best_params = params
            best_acc = acc
            best_commit_current = commit
            print(f"  *** NEW BEST: {params} params @ {acc:.4f} ***")
        elif status == "keep":
            # Keep is above 98% but not fewer params — reset
            subprocess.run(["git", "reset", "--hard", best_commit_current], capture_output=True)

    print(f"\n=== GOAL B COMPLETE ===")
    print(f"Best: {best_params} params @ {best_acc:.4f} (commit {best_commit_current})")
