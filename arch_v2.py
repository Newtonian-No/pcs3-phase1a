"""
PC-S³ v2: Error-Driven State Space Architectures (patch-based)
===============================================================
All variants use ViT-style patch embedding → Mamba scans 64-token sequences.

Variants:
  dual_gate    — Error-driven forget/update gates
  dual_stream  — Content SSM + Error SSM + cross-attention
  combo        — Dual-gate + dual-stream combined
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm import SelectiveSSM, PatchEmbed


# ═══════════════════════════════════════════════════════════════
# Shared: Self-Prediction Error Generator
# ═══════════════════════════════════════════════════════════════

class SelfPredictor(nn.Module):
    """Predict input from hidden state → compute prediction error."""
    def __init__(self, d_model: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h, x):
        """h, x: (B, L, D) → error: (B, L, D)"""
        return x - self.predictor(h)


# ═══════════════════════════════════════════════════════════════
# Variant 1: Error-Driven Dual-Gate
# ═══════════════════════════════════════════════════════════════

class DualGateSSM(nn.Module):
    """
    Error-driven forget/update gates replace standard state transition.

    h_new = forget·h + (1-forget)·update·h
    forget = σ(W_f·error), update = tanh(W_u·error)
    """

    def __init__(self, d_model=256, d_state=16, num_classes=100):
        super().__init__()
        self.d_model = d_model

        self.patch_embed = PatchEmbed(d_model=d_model)
        self.content_ssm = SelectiveSSM(d_model, d_state)
        self.self_pred = SelfPredictor(d_model)

        self.W_forget = nn.Linear(d_model, d_model)
        self.W_update = nn.Linear(d_model, d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = self.patch_embed(x)  # (B, L, D)

        h_content = self.content_ssm(x, mode="vanilla")  # (B, L, D)
        error = self.self_pred(h_content, x)

        forget = torch.sigmoid(self.W_forget(error))
        update = torch.tanh(self.W_update(error))
        h_final = forget * h_content + (1 - forget) * update * h_content

        return self.classifier(h_final.mean(dim=1))


# ═══════════════════════════════════════════════════════════════
# Variant 2: Dual-Stream + Error Accumulator
# ═══════════════════════════════════════════════════════════════

class DualStreamSSM(nn.Module):
    """
    Content SSM (full dim) + Error SSM (D//4) → cross-attention fusion.
    """

    def __init__(self, d_model=256, d_state=16, num_classes=100):
        super().__init__()
        d_error = d_model // 4
        d_state_err = max(4, d_state // 4)

        self.patch_embed = PatchEmbed(d_model=d_model)
        self.content_ssm = SelectiveSSM(d_model, d_state)
        self.self_pred = SelfPredictor(d_model)

        self.error_proj = nn.Linear(d_model, d_error)
        self.error_ssm = SelectiveSSM(d_error, d_state_err)
        self.attn_scale = d_error ** -0.5

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = self.patch_embed(x)

        h_content = self.content_ssm(x, mode="vanilla")
        error = self.self_pred(h_content, x)
        error_small = self.error_proj(error)
        h_error = self.error_ssm(error_small, mode="vanilla")

        attn = torch.matmul(h_error, h_content.transpose(-2, -1)) * self.attn_scale
        attn = F.softmax(attn, dim=-1)
        h_final = h_content + attn @ h_content

        return self.classifier(h_final.mean(dim=1))


# ═══════════════════════════════════════════════════════════════
# Variant 3: Combo — Dual-Gate + Dual-Stream
# ═══════════════════════════════════════════════════════════════

class ComboSSM(nn.Module):
    """Dual-stream with error-driven gate fusion."""

    def __init__(self, d_model=256, d_state=16, num_classes=100):
        super().__init__()
        d_error = d_model // 4
        d_state_err = max(4, d_state // 4)

        self.patch_embed = PatchEmbed(d_model=d_model)
        self.content_ssm = SelectiveSSM(d_model, d_state)
        self.self_pred = SelfPredictor(d_model)

        self.error_proj = nn.Linear(d_model, d_error)
        self.error_ssm = SelectiveSSM(d_error, d_state_err)
        self.error_to_gate = nn.Linear(d_error, d_model)

        self.W_forget = nn.Linear(d_model, d_model)
        self.W_update = nn.Linear(d_model, d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 3, 32, 32)
        x = self.patch_embed(x)

        h_content = self.content_ssm(x, mode="vanilla")
        error = self.self_pred(h_content, x)
        error_small = self.error_proj(error)
        h_error = self.error_ssm(error_small, mode="vanilla")

        gate_signal = self.error_to_gate(h_error)
        forget = torch.sigmoid(self.W_forget(gate_signal))
        update = torch.tanh(self.W_update(gate_signal))
        h_final = forget * h_content + (1 - forget) * update * h_content

        return self.classifier(h_final.mean(dim=1))


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def make_model_v2(mode, d_model=256, d_state=16, num_classes=100, **kw):
    models = {
        "dual_gate": DualGateSSM,
        "dual_stream": DualStreamSSM,
        "combo": ComboSSM,
    }
    if mode not in models:
        raise ValueError(f"Unknown v2 mode: {mode}")
    return models[mode](d_model=d_model, d_state=d_state, num_classes=num_classes)
