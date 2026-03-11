"""MNIST search — training script (D1: 69 pixels)."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 1200
TARGET_ACC = 0.95
MAX_EPOCHS = 200

# Top 69 ranked pixels
POSITIONS = [(13, 14), (14, 14), (16, 13), (12, 14), (15, 14), (14, 17), (15, 13), (16, 14), (17, 13), (15, 17), (12, 15), (13, 17), (14, 13), (15, 16), (16, 16), (13, 13), (13, 15), (21, 9), (14, 9), (13, 9), (19, 11), (19, 10), (17, 14), (16, 12), (17, 12), (22, 10), (20, 9), (15, 9), (20, 10), (6, 15), (11, 15), (9, 11), (14, 16), (12, 9), (5, 15), (18, 11), (11, 10), (12, 10), (13, 10), (10, 10), (12, 18), (18, 13), (21, 10), (11, 18), (6, 16), (18, 12), (18, 10), (14, 8), (12, 17), (20, 11), (6, 14), (10, 11), (23, 12), (16, 15), (21, 8), (22, 9), (19, 12), (15, 8), (20, 8), (13, 12), (22, 11), (5, 14), (13, 11), (14, 10), (19, 9), (13, 18), (14, 15), (5, 16), (15, 10)]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        rows = torch.tensor([p[0] for p in POSITIONS], dtype=torch.long)
        cols = torch.tensor([p[1] for p in POSITIONS], dtype=torch.long)
        self.register_buffer('rows', rows)
        self.register_buffer('cols', cols)
        n = len(POSITIONS)
        self.fc1 = nn.Linear(n, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x[:, 0, self.rows, self.cols]
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

t_start = time.time()
torch.manual_seed(7)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
print(f"pixel_count: {len(POSITIONS)}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

model.train()
epoch = 0
t_train_start = time.time()
best_val = 0.0
no_improve = 0

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, 128):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y, label_smoothing=0.1)
        loss.backward()
        optimizer.step()
    scheduler.step()
    training_time = time.time() - t_train_start
    acc = evaluate_accuracy(model)
    model.train()
    if epoch % 10 == 0 or acc >= TARGET_ACC:
        print(f"epoch {epoch} | val_acc: {acc:.4f} | best: {best_val:.4f} | time: {training_time:.1f}s")
    if acc > best_val:
        best_val = acc
        no_improve = 0
    else:
        no_improve += 1
    if acc >= TARGET_ACC:
        break
    if no_improve >= 25:
        break
    if training_time >= TIME_BUDGET:
        break
    if epoch >= MAX_EPOCHS:
        break

print()
training_time = time.time() - t_train_start
val_accuracy = evaluate_accuracy(model)
t_end = time.time()
print("---")
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_epochs:     {epoch}")
print(f"num_params:       {num_params}")
print(f"pixel_count:      {len(POSITIONS)}")
