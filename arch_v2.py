"""
PC-S³ v2: Error-Driven State Space Architectures
=================================================
Three novel architectures where prediction error is a first-class citizen
in state transitions — not an external plug-in.

Variants:
  dual_gate    — Error-driven forget/update gates
  dual_stream  — Content SSM + Error SSM + cross-attention
  combo        — Dual-gate + dual-stream combined

All variants: pure PyTorch, ~200 lines, reuses ssm.SelectiveSSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm import SelectiveSSM


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

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, 1, D) hidden state from content SSM
            x: (B, 1, D) original embedded input
        Returns:
            error: (B, 1, D) prediction error
        """
        x_pred = self.predictor(h)
        return x - x_pred


# ═══════════════════════════════════════════════════════════════
# Variant 1: Error-Driven Dual-Gate
# ═══════════════════════════════════════════════════════════════

class DualGateSSM(nn.Module):
    """
    Replaces standard SSM state transition with error-driven gates.

    Standard:  h_new = A·h + B·x
    Ours:      h_new = forget·h + (1-forget)·update·Bx
               forget = σ(W_f·error), update = tanh(W_u·error)

    Intuition: large error → model is surprised → forget old state, trust new input.
    """

    def __init__(self, d_model: int = 256, d_state: int = 16,
                 input_dim: int = 3072, num_classes: int = 100):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.embed = nn.Linear(input_dim, d_model)

        # Content SSM (standard Mamba, vanilla mode)
        self.content_ssm = SelectiveSSM(d_model, d_state)

        # Self-prediction error
        self.self_pred = SelfPredictor(d_model)

        # Error-driven gates
        self.W_forget = nn.Linear(d_model, d_model)
        self.W_update = nn.Linear(d_model, d_model)

        # Output
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1)
        x = self.embed(x).unsqueeze(1)  # (B, 1, D)

        # Content stream
        h_content = self.content_ssm(x, error=None, mode="vanilla")  # (B, 1, D)

        # Self-prediction error
        error = self.self_pred(h_content, x)  # (B, 1, D)

        # Error-driven gates
        forget = torch.sigmoid(self.W_forget(error))  # (B, 1, D)
        update = torch.tanh(self.W_update(error))      # (B, 1, D)

        # Gated fusion: error controls how much to keep vs replace
        h_final = forget * h_content + (1 - forget) * update * h_content

        return self.classifier(h_final.squeeze(1))


# ═══════════════════════════════════════════════════════════════
# Variant 2: Dual-Stream + Error Accumulator
# ═══════════════════════════════════════════════════════════════

class DualStreamSSM(nn.Module):
    """
    Content SSM + Error SSM with cross-attention fusion.

    The error stream has its OWN state space — it can "remember"
    which patterns have been historically hard to predict.

    Content SSM:  full dimension (D)
    Error SSM:    D//4 dimension (lightweight)
    Fusion:       cross-attention from error to content
    """

    def __init__(self, d_model: int = 256, d_state: int = 16,
                 input_dim: int = 3072, num_classes: int = 100):
        super().__init__()
        self.d_model = d_model
        d_error = d_model // 4
        d_state_err = d_state // 4

        # Input projection
        self.embed = nn.Linear(input_dim, d_model)

        # Content stream
        self.content_ssm = SelectiveSSM(d_model, d_state)

        # Self-prediction error
        self.self_pred = SelfPredictor(d_model)

        # Error projection (D → D//4)
        self.error_proj = nn.Linear(d_model, d_error)

        # Error SSM (lighter, learns error patterns over time)
        self.error_ssm = SelectiveSSM(d_error, d_state_err)

        # Cross-attention: error attends to content
        self.attn_scale = d_error ** -0.5

        # Output
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1)
        x = self.embed(x).unsqueeze(1)  # (B, 1, D)

        # Content stream
        h_content = self.content_ssm(x, error=None, mode="vanilla")  # (B, 1, D)

        # Self-prediction error
        error = self.self_pred(h_content, x)  # (B, 1, D)
        error_small = self.error_proj(error)   # (B, 1, D//4)

        # Error stream — learns to accumulate error patterns
        h_error = self.error_ssm(error_small, error=None, mode="vanilla")  # (B, 1, D//4)

        # Cross-attention: error context modulates content
        # h_error: (B, 1, D//4), h_content: (B, 1, D)
        attn = torch.matmul(h_error, h_content.transpose(-2, -1)) * self.attn_scale
        attn = F.softmax(attn, dim=-1)  # (B, 1, 1)
        h_final = h_content + attn * h_content  # residual attention modulation

        return self.classifier(h_final.squeeze(1))


# ═══════════════════════════════════════════════════════════════
# Variant 3: Combo — Dual-Gate + Dual-Stream
# ═══════════════════════════════════════════════════════════════

class ComboSSM(nn.Module):
    """
    Full combination: Dual-Stream with Error-Driven Gate fusion.

    Content SSM (D) + Error SSM (D//4) → error-driven gates → fused output.
    """

    def __init__(self, d_model: int = 256, d_state: int = 16,
                 input_dim: int = 3072, num_classes: int = 100):
        super().__init__()
        self.d_model = d_model
        d_error = d_model // 4
        d_state_err = d_state // 4

        self.embed = nn.Linear(input_dim, d_model)

        # Content stream
        self.content_ssm = SelectiveSSM(d_model, d_state)

        # Self-prediction error
        self.self_pred = SelfPredictor(d_model)

        # Error stream
        self.error_proj = nn.Linear(d_model, d_error)
        self.error_ssm = SelectiveSSM(d_error, d_state_err)

        # Error → gate projection (D//4 → D)
        self.error_to_gate = nn.Linear(d_error, d_model)

        # Error-driven gates
        self.W_forget = nn.Linear(d_model, d_model)
        self.W_update = nn.Linear(d_model, d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.flatten(1)
        x = self.embed(x).unsqueeze(1)  # (B, 1, D)

        # Content stream
        h_content = self.content_ssm(x, error=None, mode="vanilla")  # (B, 1, D)

        # Error stream
        error = self.self_pred(h_content, x)           # (B, 1, D)
        error_small = self.error_proj(error)            # (B, 1, D//4)
        h_error = self.error_ssm(error_small, error=None, mode="vanilla")  # (B, 1, D//4)

        # Project error state back to D for gating
        error_gate_signal = self.error_to_gate(h_error)  # (B, 1, D)

        # Error-driven gates
        forget = torch.sigmoid(self.W_forget(error_gate_signal))
        update = torch.tanh(self.W_update(error_gate_signal))

        # Gated fusion
        h_final = forget * h_content + (1 - forget) * update * h_content

        return self.classifier(h_final.squeeze(1))


# ═══════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════

def make_model_v2(mode: str, input_dim: int = 3072, num_classes: int = 100,
                  d_model: int = 256, d_state: int = 16):
    """Create a v2 architecture by mode name."""
    models = {
        "dual_gate": DualGateSSM,
        "dual_stream": DualStreamSSM,
        "combo": ComboSSM,
    }
    if mode not in models:
        raise ValueError(f"Unknown v2 mode: {mode}. Choose from {list(models.keys())}")
    return models[mode](d_model=d_model, d_state=d_state,
                        input_dim=input_dim, num_classes=num_classes)
