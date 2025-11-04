# app/inference_gan.py
from io import BytesIO
from pathlib import Path
import base64
import torch
from torchvision.utils import make_grid, save_image

from .gan import Generator

class GanSampler:
    def __init__(self, g_ckpt_path: str = "models/gan_G.pt"):
        ckpt = torch.load(g_ckpt_path, map_location="cpu")
        self.z_dim = ckpt.get("z_dim", 100)
        self.G = Generator(z_dim=self.z_dim)
        self.G.load_state_dict(ckpt["G"])
        self.G.eval()

    @torch.no_grad()
    def sample_png_bytes(self, n: int = 16, seed: int | None = None, nrow: int = 4) -> bytes:
        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(n, self.z_dim)
        imgs = self.G(z)  # (n,1,28,28) in [-1,1]
        grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
        buf = BytesIO()
        save_image(grid, buf, format="PNG")
        return buf.getvalue()

    @torch.no_grad()
    def sample_base64(self, n: int = 16, seed: int | None = None, nrow: int = 4) -> str:
        png_bytes = self.sample_png_bytes(n=n, seed=seed, nrow=nrow)
        return base64.b64encode(png_bytes).decode("utf-8")
