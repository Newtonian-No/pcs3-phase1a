"""
PC-S³ Models — Patch-based SSM for image classification.
Replaces flat MLP input with ViT-style patch embedding → Mamba can actually scan.

V1 modes: vanilla / concat / delta / B / C
V2 modes: dual_gate / dual_stream / combo (in arch_v2.py)
"""

import torch
import torch.nn as nn
from ssm import MambaBlock, PatchEmbed


class PCNClassifier(nn.Module):
    """
    Patch-based SSM classifier — replaces old flat-input PCN.

    Architecture:
        Image (3,32,32) → PatchEmbed → (B, 64, D) tokens
        → MambaBlock → (B, 64, D)
        → GlobalAvgPool + Classifier → (B, 100)

    Modes:
        vanilla  — no error modulation
        concat   — error added to SSM input
        delta    — error modulates Δ
        B        — error modulates B
        C        — error modulates C
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 d_model=256, d_state=16, num_classes=100,
                 mode="vanilla", n_iter=1):
        super().__init__()
        self.mode = mode
        self.n_iter = n_iter

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_channels=in_channels, d_model=d_model,
        )

        self.mamba = MambaBlock(d_model, d_state)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # Input: (B, 3, 32, 32) or (B, 3072) — auto-detect
        if x.dim() == 2:
            # Legacy flat input — reshape to image
            x = x.view(-1, 3, 32, 32)

        x = self.patch_embed(x)  # (B, L, D) where L=64

        # Mamba processes sequence
        error = None
        if self.mode != "vanilla":
            error = torch.zeros_like(x)

        out = self.mamba(x, error=error, mode=self.mode)  # (B, L, D)

        # Global average pooling over patches
        out = out.mean(dim=1)  # (B, D)

        return self.classifier(out)


def make_model(mode="vanilla", input_dim=3072, num_classes=100,
               d_model=256, num_layers=3, n_iter=1):
    """Factory for v1 models (patch-based)."""
    return PCNClassifier(
        d_model=d_model, d_state=16, num_classes=num_classes,
        mode=mode, n_iter=n_iter,
    )
