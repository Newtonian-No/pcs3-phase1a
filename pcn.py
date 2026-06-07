"""
PC-S³ Models — Patch-based SSM for image classification.

V1 modes: vanilla / concat / delta / B / C / concat_shuffled
V2 modes: dual_gate / dual_stream / combo (in arch_v2.py)
"""

import torch
import torch.nn as nn
from ssm import MambaBlock, PatchEmbed


class SelfPredictor(nn.Module):
    """Predict input from hidden state → compute prediction error."""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h, x):
        return x - self.net(h)


class PCNClassifier(nn.Module):
    """
    Patch-based SSM classifier with self-prediction error.

    Architecture:
        Image → PatchEmbed → (B, L, D)
        → MambaBlock (with optional error modulation)
        → SelfPredictor → error = x - pred(h)
        → GlobalAvgPool + Classifier

    Modes:
        vanilla          — no error
        concat           — error added to SSM input
        concat_shuffled  — error shuffled across batch (control)
        delta            — error modulates Δ
        B                — error modulates B
        C                — error modulates C
    """

    def __init__(self, d_model=256, d_state=16, num_classes=100,
                 mode="vanilla"):
        super().__init__()
        self.mode = mode

        self.patch_embed = PatchEmbed(d_model=d_model)
        self.mamba = MambaBlock(d_model, d_state)
        self.self_pred = SelfPredictor(d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)

        x = self.patch_embed(x)  # (B, L, D)

        # First pass: vanilla (no error) to get hidden state
        h = self.mamba(x, error=None, mode="vanilla")

        # Compute self-prediction error
        error = self.self_pred(h, x)  # (B, L, D)

        # For concat_shuffled: destroy per-sample error semantics
        if self.mode == "concat_shuffled":
            idx = torch.randperm(error.size(0), device=error.device)
            error = error[idx]

        # Second pass: with error modulation
        if self.mode == "vanilla":
            out = h  # reuse first pass output
        else:
            out = self.mamba(x, error=error, mode=self.mode)

        out = out.mean(dim=1)  # (B, D)
        return self.classifier(out)


def make_model(mode="vanilla", input_dim=3072, num_classes=100,
               d_model=256, num_layers=3, n_iter=1):
    return PCNClassifier(
        d_model=d_model, d_state=16, num_classes=num_classes,
        mode=mode,
    )
