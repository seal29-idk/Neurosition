import torch
import torch.nn as nn


class CoordConv(nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
        return torch.cat([x, grid], dim=1)


class NeuroRenderer(nn.Module):
    def __init__(self, coord_dim=2, latent=256, base_ch=128, out_size=128):
        super().__init__()
        self.out_size = out_size

        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, latent), nn.ReLU(),
        )

        self.fc = nn.Linear(latent, base_ch * 8 * 4 * 4)

        self.up = nn.Sequential(
            self._block(base_ch * 8, base_ch * 4),
            self._block(base_ch * 4, base_ch * 2),
            self._block(base_ch * 2, base_ch),
            self._block(base_ch, base_ch),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.base_ch = base_ch

    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, coords):
        z = self.mlp(coords)
        h = self.fc(z).view(-1, self.base_ch * 8, 4, 4)
        return self.up(h)
