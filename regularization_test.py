import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.integrate import simpson as simps

# --- Загрузка данных ---
df = pd.read_csv("credit.csv")
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).unsqueeze(1)
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# --- Гиперпараметры ---
hidden_layers = [32, 32]
learning_rate = 1e-3
batch_size = 64
patience = 10
max_epochs = 10
epsilon_convergence = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Модель ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate=0.0, use_batch_norm=False):
        super().__init__()
        layers = []
        in_features = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# --- Конфигурации регуляризации ---
configs = {
    "No Reg":         {"dropout": 0.0, "batchnorm": False, "l2": 0.0},
    "Dropout":        {"dropout": 0.3, "batchnorm": False, "l2": 0.0},
    "BatchNorm":      {"dropout": 0.0, "batchnorm": True,  "l2": 0.0},
    "L2 Regularizer": {"dropout": 0.0, "batchnorm": False, "l2": 1e-4},
}

results = {}

# --- Основной цикл по конфигурациям ---
for name, cfg in configs.items():
    print(f"\n--- Training with {name} ---")
    model = MLP(X.shape[1], 1, hidden_layers, cfg["dropout"], cfg["batchnorm"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg["l2"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    batch_mse_per_epoch = []

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in tqdm(range(max_epochs), desc=f"Training {name}"):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_losses = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            train_loss += loss.item() * X_batch.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        train_losses.append(train_loss / total)
        train_accs.append(correct / total)
        batch_mse_per_epoch.append(np.var(batch_losses))

        # Валидация
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        val_preds = []
        val_targets = []
        val_proba = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_proba.append(torch.sigmoid(outputs).cpu().numpy())
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_losses.append(val_loss / total)
        val_accs.append(correct / total)

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Метрики после обучения
    model.load_state_dict(best_model_state)
    val_preds = np.vstack(val_preds).flatten()
    val_targets = np.vstack(val_targets).flatten()
    val_proba = np.vstack(val_proba).flatten()
    precision = precision_score(val_targets, val_preds)
    recall = recall_score(val_targets, val_preds)
    auc_pr = average_precision_score(val_targets, val_preds)
    aulc = simps(val_losses, dx=1)

    # Сходимость
    convergence_epoch = 0
    for i in range(1, len(val_losses)):
        if abs(val_losses[i-1] - val_losses[i]) < epsilon_convergence:
            convergence_epoch = i
            break

    results[name] = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "precision": precision,
        "recall": recall,
        "auc_pr": auc_pr,
        "aulc": aulc,
        "convergence_epoch": convergence_epoch,
        "batch_mse": np.mean(batch_mse_per_epoch),
        "val_proba": val_proba,
        "val_targets": val_targets
    }

# --- Построение графиков ---
plt.figure(figsize=(14, 6))
for name in configs:
    plt.plot(results[name]["val_loss"], label=f"{name}")
plt.title("Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig('reg_loss.png')

plt.figure(figsize=(14, 6))
for name in configs:
    precision, recall, _ = precision_recall_curve(results[name]["val_targets"], results[name]["val_proba"])
    plt.plot(recall, precision, label=f"{name} (AUC-PR={results[name]['auc_pr']:.3f})")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.savefig('reg_PR.png')

# --- Табличный вывод ---
print("\n=== Сравнение конфигураций ===")
for name in configs:
    r = results[name]
    print(f"\n{name}:")
    print(f"  Precision         = {r['precision']:.4f}")
    print(f"  Recall            = {r['recall']:.4f}")
    print(f"  AUC-PR            = {r['auc_pr']:.4f}")
    print(f"  AULC              = {r['aulc']:.4f}")
    print(f"  Convergence Epoch = {r['convergence_epoch']}")
    print(f"  Batch Loss MSE    = {r['batch_mse']:.6f}")
