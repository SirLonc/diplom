import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve
from scipy.integrate import simpson as simps

# Функции потерь
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class CyclicalFocalLoss(nn.Module):
    def __init__(self, gamma_list, reduction='mean'):
        super().__init__()
        self.gamma_list = gamma_list
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, inputs, targets):
        gamma = self.gamma_list[self.epoch % len(self.gamma_list)]
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = (1 - pt) ** gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Модель
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Обучение
from tqdm import tqdm

def train_model(name, loss_fn):
    model = MLP(input_dim, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    pr_curves = []

    best_val_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0
    batch_mse_per_epoch = []

    for epoch in tqdm(range(max_epochs), desc="Training epochs"):
        if isinstance(loss_fn, CyclicalFocalLoss):
            loss_fn.set_epoch(epoch)

        model.train()
        epoch_losses = []
        correct, total = 0, 0

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_losses.append(np.mean(epoch_losses))
        train_accs.append(correct / total)
        batch_mse_per_epoch.append(np.var(epoch_losses))

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = loss_fn(outputs, yb)
                val_loss += loss.item() * xb.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_preds.extend(probs.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_loss /= total
        val_losses.append(val_loss)
        val_accs.append(correct / total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_model_state)

    val_preds = np.array(val_preds).flatten()
    val_targets = np.array(val_targets).flatten()

    precision = precision_score(val_targets, val_preds > 0.5)
    recall = recall_score(val_targets, val_preds > 0.5)
    auc_pr = average_precision_score(val_targets, val_preds)
    precision_curve, recall_curve, _ = precision_recall_curve(val_targets, val_preds)
    aulc = simps(val_losses, dx=1)
    batch_var = np.mean(batch_mse_per_epoch)

    return {
        'label': name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'precision': precision,
        'recall': recall,
        'auc_pr': auc_pr,
        'aulc': aulc,
        'batch_var': batch_var,
        'pr_curve': (precision_curve, recall_curve)
    }

# --- Данные ---
df = pd.read_csv("credit.csv")
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)
X = StandardScaler().fit_transform(X)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# --- Настройки ---
input_dim = X.shape[1]
hidden_layers = [32, 32]
learning_rate = 1e-4
max_epochs = 10
patience = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Запуск ---
results = [
    train_model("BCE", nn.BCEWithLogitsLoss()),
    train_model("Focal γ=1", FocalLoss(gamma=1)),
    train_model("Focal γ=2", FocalLoss(gamma=2)),
    train_model("Focal γ=5", FocalLoss(gamma=5)),
    train_model("Cyclical Focal", CyclicalFocalLoss([1, 2, 5]))
]

# --- Графики ---
plt.figure(figsize=(14, 6))
for r in results:
    plt.plot(r['val_losses'], label=r['label'])
plt.title("Validation Loss by Loss Function")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('loss_loss.png')

plt.figure(figsize=(14, 6))
for r in results:
    plt.plot(r['val_accs'], label=r['label'])
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('acc_loss.png')

plt.figure(figsize=(10, 8))
for r in results:
    prec, rec = r['pr_curve']
    plt.plot(rec, prec, label=f"{r['label']} (AUC-PR={r['auc_pr']:.3f})")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.savefig('PR_loss.png')

# --- Вывод ---
for r in results:
    print(f"--- {r['label']} ---")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall: {r['recall']:.4f}")
    print(f"AUC-PR: {r['auc_pr']:.4f}")
    print(f"AULC (Loss Curve Area): {r['aulc']:.4f}")
    print(f"Batch Loss Variance: {r['batch_var']:.6f}")
    print()
