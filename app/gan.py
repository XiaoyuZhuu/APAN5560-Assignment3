# app/gan.py
import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            # After reshape: (B, 128, 7, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # -> (B,1,28,28)
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (B,100)
        x = self.fc(z)                           # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)               # (B, 128, 7, 7)
        x = self.net(x)                          # (B, 1, 28, 28)
        return x


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   # -> (B,64,14,14)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (B,128,7,7)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)  # output logit

    def forward(self, x):
        f = self.features(x)
        f = f.view(x.size(0), -1)
        logit = self.classifier(f)
        return logit  
