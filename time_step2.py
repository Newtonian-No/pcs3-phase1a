#!/usr/bin/env python3
"""Quick timing of Step 2 architecture on 5090."""
import torch, torch.nn as nn, time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ═══ Step 2 Architecture ═══

class ConvStem(nn.Module):
    def __init__(self, in_c=3, dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.GELU(),
        )
    def forward(self, x):
        return self.stem(x)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_c=128, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(in_c, d_model, patch_size, patch_size)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.proj(x)  # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        return self.norm(x)

class MambaBlock(nn.Module):
    """Simplified Mamba block for timing — matched compute profile."""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.d_inner = d_inner

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :x.size(-1)]
        x = nn.functional.silu(x)
        x = x.transpose(1, 2)
        x = x * nn.functional.silu(z)
        x = self.out_proj(x)
        return residual + x

class PCNLayer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 512), nn.GELU(),
            nn.Linear(512, d_model),
        )
        self.error_norm = nn.LayerNorm(d_model)
        self.concat_proj = nn.Linear(d_model * 2, d_model)
    def forward(self, x):
        pred = self.predictor(x)
        error = self.error_norm(x - pred)
        z = self.concat_proj(torch.cat([nn.functional.layer_norm(x, x.shape[-1:]), error], dim=-1))
        return z

class Step2Model(nn.Module):
    def __init__(self, d_model=256, n_layers=12, num_classes=100):
        super().__init__()
        self.stem = ConvStem(3, 128)
        self.patch = PatchEmbed(2, 128, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.pcn = nn.ModuleList([PCNLayer(d_model) for _ in range(3)])
        self.cls_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.n_layers = n_layers

    def forward(self, x):
        x = self.stem(x)       # (B, 128, 32, 32)
        x = self.patch(x)      # (B, 256, 256)
        for i in range(self.n_layers):
            pc_idx = i * 3 // self.n_layers
            if i % 4 == 0 and pc_idx < 3:
                x = self.pcn[pc_idx](x)
            x = self.blocks[i](x)
        x = x.mean(dim=1)
        return self.head(self.cls_norm(x))

# ═══ Timing ═══

m = Step2Model(d_model=256, n_layers=12).cuda()
params = sum(p.numel() for p in m.parameters())
print(f"Params: {params:,}")

opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
x = torch.randn(128, 3, 32, 32).cuda()
y = torch.randint(0, 100, (128,)).cuda()

# Warmup
for _ in range(5):
    m(x).sum().backward()
    opt.step(); opt.zero_grad()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    out = m(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step(); opt.zero_grad()
torch.cuda.synchronize()
t1 = time.time()

per_iter = (t1-t0)/20
per_epoch = per_iter * 50000/128 / 60
print(f"Per iter: {per_iter*1000:.0f}ms | Per epoch: {per_epoch:.1f} min")
print(f"300 epochs: {per_epoch*300/60:.1f} hours")
