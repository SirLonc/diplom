import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np



class GeneratorClass(nn.Module):
    def __init__(self, noise_dim=100, output_dim=2):  
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        return self.model(z)


class DiscriminatorClass(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        return self.model(x)


def train_gan(generator, discriminator, X, y, epochs=100, batch_size=64, noise_dim=100, n_critic=5, clip_value=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    minority_data = X[y == 1].float().to(device)
    dataloader = DataLoader(TensorDataset(minority_data), batch_size=batch_size, shuffle=True)

    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=5e-5)
    optimizer_G = optim.RMSprop(generator.parameters(), lr=5e-5)

    for epoch in range(epochs):
        for i, (real_data,) in enumerate(dataloader):
            real_data = real_data.to(device)

            # === Обновление дискриминатора ===
            for _ in range(n_critic):
                z = torch.randn(real_data.size(0), noise_dim).to(device)
                fake_data = generator(z).detach()

                d_loss = -(torch.mean(discriminator(real_data)) - torch.mean(discriminator(fake_data)))

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Обновление генератора ===
            z = torch.randn(real_data.size(0), noise_dim).to(device)
            fake_data = generator(z)
            g_loss = -torch.mean(discriminator(fake_data))

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

        if (epoch + 1) % 10 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")



def generate_target_samples(generator, n_samples, noise_dim=100):
    generator.eval()
    device = next(generator.parameters()).device
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim).to(device)
        samples = generator(z).cpu().numpy()
    return samples