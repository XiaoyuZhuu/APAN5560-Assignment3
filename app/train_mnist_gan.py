# app/train_mnist_gan.py
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils

from .gan import Generator, Discriminator

def save_grid(samples, out_path: Path, nrow=8):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(samples, out_path, nrow=nrow, normalize=True, value_range=(-1, 1))

def get_loaders(batch_size=128, num_workers=2):
    tfm = transforms.Compose([
        transforms.ToTensor(),                # [0,1]
        transforms.Normalize((0.5,), (0.5,))  # [-1,1]
    ])
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    train = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader

def train(
    epochs=20,
    z_dim=100,
    batch_size=128,
    lr=2e-4,
    beta1=0.5,
    beta2=0.999,
    sample_every=1,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loader = get_loaders(batch_size=batch_size)
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)

    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()
    g_opt = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))
    d_opt = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))

    out_models = Path("models"); out_models.mkdir(exist_ok=True)
    out_imgs = Path("outputs"); out_imgs.mkdir(exist_ok=True)

    fixed_z = torch.randn(64, z_dim, device=device)

    step = 0
    for epoch in range(1, epochs + 1):
        for real, _ in loader:
            real = real.to(device)  # (B,1,28,28)
            b = real.size(0)

            
            # Train Discriminator
            z = torch.randn(b, z_dim, device=device)
            with torch.no_grad():
                fake = G(z)

            D.train(); G.train()
            d_opt.zero_grad()

            real_logit = D(real)
            fake_logit = D(fake)

            # Real labels = 1, Fake labels = 0
            real_labels = torch.ones_like(real_logit)
            fake_labels = torch.zeros_like(fake_logit)

            d_loss_real = criterion(real_logit, real_labels)
            d_loss_fake = criterion(fake_logit, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_opt.step()

            # Train Generator
            z = torch.randn(b, z_dim, device=device)
            g_opt.zero_grad()
            gen = G(z)
            gen_logit = D(gen)
            g_loss = criterion(gen_logit, torch.ones_like(gen_logit))  # want D(gen)=1
            g_loss.backward()
            g_opt.step()

            step += 1

        print(f"[Epoch {epoch}] D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

        # Save sample grid
        if epoch % sample_every == 0:
            with torch.no_grad():
                G.eval()
                samples = G(fixed_z).cpu()
            save_grid(samples, out_imgs / f"samples_epoch_{epoch:03d}.png", nrow=8)

        # Save checkpoints (keep generator for inference)
        torch.save({"G": G.state_dict(), "z_dim": z_dim}, out_models / "gan_G.pt")
        torch.save({"D": D.state_dict()}, out_models / "gan_D.pt")

    print("Training finished. Checkpoints in models/, samples in outputs/")

if __name__ == "__main__":
    epochs = int(os.getenv("EPOCHS", "20"))
    train(epochs=epochs)
