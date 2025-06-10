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
from imblearn.over_sampling import SMOTE
from wgan import GeneratorClass, DiscriminatorClass, train_gan, generate_target_samples

# Используем вашу архитектуру MLP
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

def train_model(X_train, y_train, label='original'):
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = MLP(input_dim, output_dim, hidden_layers, use_batch_norm=False).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_losses, val_losses = [], []
    train_batch_losses = []

    val_preds_collect, val_targets_collect = [], []
    val_probs_collect = []

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in tqdm(range(max_epochs), desc="Training epochs"):
        model.train()
        total_loss = 0
        batch_losses_epoch = []

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_losses_epoch.append(loss.item())

        train_losses.append(total_loss / len(train_loader))
        train_batch_losses.append(batch_losses_epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                probs = torch.sigmoid(outputs)

                val_preds.append(preds.cpu().numpy())
                val_probs.append(probs.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        val_preds_collect = np.vstack(val_preds).flatten()
        val_probs_collect = np.vstack(val_probs).flatten()
        val_targets_collect = np.vstack(val_targets).flatten()

        # Early Stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_model_state)

    # Метрики
    precision = precision_score(val_targets_collect, val_preds_collect)
    recall = recall_score(val_targets_collect, val_preds_collect)
    auc_pr = average_precision_score(val_targets_collect, val_probs_collect)
    aulc = np.trapz(val_losses)
    batch_mse = np.mean([np.var(epoch_losses) for epoch_losses in train_batch_losses])

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'precision': precision,
        'recall': recall,
        'auc_pr': auc_pr,
        'aulc': aulc,
        'batch_mse': batch_mse,
        'probs': val_probs_collect,
        'targets': val_targets_collect,
        'batch_losses': train_batch_losses,
        'label': label
    }

# --- Данные (здесь примерные, свои подставь) ---
input_dim = 30
output_dim = 1
hidden_layers = [32, 32]
learning_rate = 1e-3
batch_size = 64
max_epochs = 10
patience = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv("credit.csv")  # ← замените на свой файл

# Предположим, что последний столбец — целевая переменная (y)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

input_dim = X.shape[1]  # автоопределение размерности входа

# --- Оригинальный датасет ---
results = []
results.append(train_model(X, y, label='Original'))

# --- SMOTE ---
sm = SMOTE()
X_smote, y_smote = sm.fit_resample(X, y)
results.append(train_model(X_smote, y_smote, label='SMOTE'))

# --- WGAN ---
# Тренируем WGAN (ты подставишь свою реализацию)
noise_dim = 100
generator = GeneratorClass(noise_dim=noise_dim, output_dim=input_dim)
discriminator = DiscriminatorClass(input_dim=input_dim)

train_gan(generator, discriminator, torch.tensor(X), torch.tensor(y), epochs=200, batch_size=64, noise_dim=noise_dim)

# Генерация сэмплов только для минорного класса
num_minority = np.sum(y == 1) * 50
gen_samples = generate_target_samples(generator, num_minority, noise_dim)
gen_labels = np.ones(len(gen_samples))

# Объединяем
X_wgan = np.vstack([X, gen_samples])
y_wgan = np.concatenate([y, gen_labels])

results.append(train_model(X_wgan, y_wgan, label='WGAN'))

# --- Визуализация ---
plt.figure(figsize=(14, 6))
for res in results:
    plt.plot(res['val_losses'], label=res['label'])
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("aug_loss.png")

plt.figure(figsize=(10, 7))
for res in results:
    precision, recall, _ = precision_recall_curve(res['targets'], res['probs'])
    plt.plot(recall, precision, label=f"{res['label']} (AUC-PR={res['auc_pr']:.3f})")

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.savefig("aug_PR.png")

# --- Печать метрик ---
for res in results:
    # AULC (Area under loss curve)
    aulc = np.trapz(res['val_losses'])

    # Batch Loss Variance
    batch_mse = np.mean([
        np.var(epoch_losses) for epoch_losses in res['batch_losses']
    ])

    print(f"--- {res['label']} ---")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall: {res['recall']:.4f}")
    print(f"AUC-PR: {res['auc_pr']:.4f}")
    print(f"AULC (Loss decay speed): {aulc:.4f}")
    print(f"Batch Loss Variance (Stability): {batch_mse:.6f}")
    print()
