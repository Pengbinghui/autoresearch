"""MNIST search — training script (D2-90: d=4, [256,128], no-act proj, seed=0, long)."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 600
TARGET_ACC = 0.90
MAX_EPOCHS = 200

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(784, 4)
        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.proj(x)  # no activation after projection
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

t_start = time.time()
torch.manual_seed(0)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
d = 4
print(f"Parameters: {num_params:,}")
print(f"d: {d}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
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
    if no_improve >= 30:
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
print(f"pixel_count:      {d}")
