"""
PC-S³ Step 2: Deeper architecture with conv stem + 256 tokens.
Reuses MambaBlock from ssm.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm import MambaBlock


class ConvStem(nn.Module):
    """Light conv stem for CIFAR-100 local texture."""
    def __init__(self, in_c=3, out_c=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.GELU(),
        )
    def forward(self, x):
        return self.stem(x)


class PatchEmbed2(nn.Module):
    """Patch embed with configurable patch size and input channels."""
    def __init__(self, img_size=32, patch_size=2, in_c=128, d_model=256):
        super().__init__()
        self.proj = nn.Conv2d(in_c, d_model, patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class SelfPredictor(nn.Module):
    """Generalized self-predictor with configurable error order K.

    K=1: scalar error e = x - x̂  (original SelfPredictor)
    K=2: generalized error ẽ = [x - x̂, x' - x̂']  (pos + velocity)
         where x' is finite-difference velocity and x̂' = vel_head(h)
    """
    def __init__(self, d_model, K=1):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        if K >= 2:
            self.vel_head = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h, x):
        # Position error (always computed)
        pos_error = x - self.net(h)

        if self.K == 1:
            return pos_error

        # Velocity error: finite-diff ground truth vs predicted
        # x' ≈ x[t] - x[t-1], pad first position as zeros
        x_vel = F.pad(x[:, 1:, :] - x[:, :-1, :], (0, 0, 0, 1))
        pred_vel = self.vel_head(h)
        vel_error = x_vel - pred_vel

        return torch.cat([pos_error, vel_error], dim=-1)  # (B, L, K*D)


class Step2Model(nn.Module):
    """
    PC-S³ Step 2 Architecture.

    ConvStem → PatchEmbed(patch_size) → N× MambaBlock (with PCN error injection)

    Args:
        use_conv_stem: If False, skip ConvStem and feed raw pixels to PatchEmbed.
    """
    def __init__(self, d_model=256, d_state=16, n_layers=12,
                 patch_size=2, pcn_interval=4, num_classes=100, mode="vanilla",
                 use_conv_stem=True):
        super().__init__()
        self.mode = mode
        self.n_layers = n_layers
        self.pcn_interval = pcn_interval
        self.use_conv_stem = use_conv_stem

        # K: error order — 1 for scalar, 2 for generalized (pos+vel)
        self.K = 2 if mode in ("gen_error", "gen_error_shuffled") else 1

        if use_conv_stem:
            self.stem = ConvStem(3, 128)
            patch_in_c = 128
        else:
            self.stem = None
            patch_in_c = 3

        self.patch = PatchEmbed2(img_size=32, patch_size=patch_size,
                                 in_c=patch_in_c, d_model=d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, K=self.K) for _ in range(n_layers)])
        self.predictors = nn.ModuleList([SelfPredictor(d_model, K=self.K) for _ in range(3)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)

        if self.use_conv_stem:
            x = self.stem(x)
        x = self.patch(x)  # (B, N_patches, d_model)

        # Phase 1: vanilla forward, collecting errors at PCN intervals
        pcn_idx = 0
        snapshots = []
        for i in range(self.n_layers):
            x = self.blocks[i](x, error=None, mode="vanilla")
            if (i + 1) % self.pcn_interval == 0 and pcn_idx < 3:
                snapshots.append(x.clone())
                pcn_idx += 1

        # Compute PCN errors
        if self.mode == "vanilla":
            out = x.mean(dim=1)
            return self.classifier(out)

        # Re-inject prediction error into later blocks
        pcn_idx = 0
        for i in range(self.n_layers):
            if (i + 1) % self.pcn_interval == 0 and pcn_idx < 3:
                h = snapshots[pcn_idx]
                error = self.predictors[pcn_idx](h, x)

                if self.mode in ("concat_shuffled", "gen_error_shuffled"):
                    idx = torch.randperm(error.size(0), device=error.device)
                    error = error[idx]

                # Map gen_error modes to concat SSM path
                ssm_mode = "concat" if self.mode in ("gen_error", "gen_error_shuffled") else self.mode
                x = self.blocks[i](x, error=error, mode=ssm_mode)
                pcn_idx += 1
            else:
                x = self.blocks[i](x, error=None, mode="vanilla")

        out = x.mean(dim=1)
        return self.classifier(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for mode in ["vanilla", "concat", "gen_error", "gen_error_shuffled"]:
        m = Step2Model(mode=mode)
        x = torch.randn(4, 3, 32, 32)
        y = m(x)
        print(f"{mode}: params={count_params(m):,}, output={y.shape}")
