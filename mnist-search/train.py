"""MNIST search — training script (D2: d=7 baseline)."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 120
TARGET_ACC = 0.95

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(784, 7)
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

t_start = time.time()
torch.manual_seed(3)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)

model.train()
epoch = 0
t_train_start = time.time()
best_val = 0.0
no_improve = 0

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, 32):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    training_time = time.time() - t_train_start
    acc = evaluate_accuracy(model)
    model.train()
    print(f"epoch {epoch} | val_acc: {acc:.4f} | time: {training_time:.1f}s")
    if acc > best_val:
        best_val = acc
        no_improve = 0
    else:
        no_improve += 1
    if acc >= TARGET_ACC:
        break
    if no_improve >= 8:
        break
    if training_time >= TIME_BUDGET:
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
print(f"projection_dim:   7")
