import torch
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, average_precision_score, precision_recall_curve

# --- Конфигурация ---
input_dim = 30
output_dim = 1
hidden_layers = [128, 64]
batch_size = 32
learning_rate = 5e-5
max_epochs = 200
patience = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# --- MLP-модель ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, use_batch_norm=False):
        super(MLP, self).__init__()
        layers = []
        in_features = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

# --- Обучение модели ---
def train_model(X, y, optimizer_name='adam'):
    dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float().unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MLP(input_dim, output_dim, hidden_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown optimizer")

    train_losses, val_losses, train_batch_losses = [], [], []

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in tqdm(range(max_epochs), desc="Training epochs"):
        model.train()
        batch_losses = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        train_batch_losses.append(batch_losses)

        # Валидация
        model.eval()
        val_loss = 0
        val_preds, val_probs, val_targets = [], [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                val_probs.append(probs.cpu().numpy())
                val_preds.append(preds.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)

    val_probs = np.vstack(val_probs).flatten()
    val_preds = np.vstack(val_preds).flatten()
    val_targets = np.vstack(val_targets).flatten()

    precision = precision_score(val_targets, val_preds)
    recall = recall_score(val_targets, val_preds)
    auc_pr = average_precision_score(val_targets, val_probs)
    aulc = np.trapz(val_losses)
    batch_mse = np.mean([np.var(ep) for ep in train_batch_losses])

    return {
        'val_losses': val_losses,
        'precision': precision,
        'recall': recall,
        'auc_pr': auc_pr,
        'aulc': aulc,
        'batch_mse': batch_mse,
        'probs': val_probs,
        'targets': val_targets,
        'label': optimizer_name.upper()
    }

# --- Загрузка данных ---
df = pd.read_csv("diabetes.csv")  # ← замените на свой файл

# Предположим, что последний столбец — целевая переменная (y)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

input_dim = X.shape[1]  # автоопределение размерности входа

# --- Обучение для разных оптимизаторов ---
results = []
for opt in ['sgd', 'rmsprop', 'adam']:
    results.append(train_model(X, y, optimizer_name=opt))

# --- Графики потерь ---
plt.figure(figsize=(10, 6))
for res in results:
    plt.plot(res['val_losses'], label=res['label'])
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig('losses_aug.png')

# --- PR-кривые ---
plt.figure(figsize=(10, 6))
for res in results:
    precision, recall, _ = precision_recall_curve(res['targets'], res['probs'])
    plt.plot(recall, precision, label=f"{res['label']} (AUC-PR={res['auc_pr']:.3f})")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.savefig('PR_aug.png')

# --- Метрики ---
for res in results:
    print(f"--- {res['label']} ---")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall: {res['recall']:.4f}")
    print(f"AUC-PR: {res['auc_pr']:.4f}")
    print(f"AULC (Loss decay speed): {res['aulc']:.4f}")
    print(f"Batch Loss Variance (Stability): {res['batch_mse']:.6f}")
    print()
