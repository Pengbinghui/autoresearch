"""MNIST search — training script."""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import TIME_BUDGET, get_data, make_batches, evaluate_accuracy

TIME_BUDGET = 120

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv3 = nn.Conv2d(6, 7, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(7 * 3 * 3, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

t_start = time.time()
torch.manual_seed(42)
model = Net()
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")
train_images, train_labels, _, _ = get_data()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

model.train()
epoch = 0
step = 0
best_acc = 0.0
patience = 3
no_improve = 0
t_train_start = time.time()

while True:
    epoch += 1
    for x, y in make_batches(train_images, train_labels, 64):
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
    acc = evaluate_accuracy(model)
    model.train()
    print(f" | val_acc: {acc:.4f}")
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
print(f"val_accuracy:     {val_accuracy:.6f}")
print(f"training_seconds: {training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"total_epochs:     {epoch}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
