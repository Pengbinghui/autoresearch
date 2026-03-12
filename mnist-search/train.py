"""MNIST search — training script (D1-90: 31 pixels, drop (11,13) least imp)."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 300
TARGET_ACC = 0.90
MAX_EPOCHS = 200

# 31 pixels — removed (11,13) from 32-set (least important, imp=48.89)
POSITIONS = [(10, 14), (10, 13), (11, 14), (13, 14), (17, 10), (9, 17), (13, 12), (18, 10), (18, 12), (12, 13), (15, 14), (19, 12), (8, 14), (12, 11), (9, 13), (11, 12), (17, 18), (16, 19), (19, 11), (9, 16), (16, 11), (14, 9), (10, 17), (12, 15), (14, 19), (11, 9), (15, 13), (8, 15), (14, 14), (16, 13), (14, 17)]

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
