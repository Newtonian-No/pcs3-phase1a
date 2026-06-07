"""
Minimal Selective State Space Model (Mamba) for PC-S³.
Pure PyTorch — no mamba-ssm dependency needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ═══════════════════════════════════════════════════════════════
# Patch Embedding (ViT-style)
# ═══════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """
    Convert image to sequence of patch tokens — enables Mamba to scan.

    Input:  (B, 3, 32, 32)
    Output: (B, num_patches, d_model)

    CIFAR-100: 32×32 → 4×4 patches → 8×8 grid = 64 tokens
    Each token: 4×4×3 = 48 dimensions → projected to d_model
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3, d_model=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)                      # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, D)
        x = x + self.pos_embed
        return x


# ═══════════════════════════════════════════════════════════════
# Selective SSM
# ═══════════════════════════════════════════════════════════════

class SelectiveSSM(nn.Module):
    """
    Single-channel selective SSM with error-gated Δ modulation.
    
    Standard Mamba:  Δ = softplus(W_Δ @ x + b_Δ)
    PC-S³ variants:  Δ modulated by prediction error e
    
    This is a per-channel SSM — for multi-channel, use MambaBlock below.
    """
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        dt_rank = dt_rank or d_model
        
        # Standard Mamba projections
        self.W_dt = nn.Linear(d_model, dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model)
        self.W_B = nn.Linear(d_model, d_state, bias=False)
        self.W_C = nn.Linear(d_model, d_state, bias=False)
        
        # Error modulation (only used in error-gated variants)
        self.W_e_dt = nn.Linear(d_model, dt_rank, bias=False)  # error → Δ
        self.W_e_B = nn.Linear(d_model, d_state, bias=False)   # error → B
        self.W_e_C = nn.Linear(d_model, d_state, bias=False)   # error → C
        
        # Concat baseline: larger projection for [x; error]
        self.W_dt_concat = nn.Linear(d_model * 2, dt_rank, bias=False)
        
        # Convolution for local context (standard Mamba conv1d)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, 
                                 padding=3, groups=d_model)
        self.act = nn.SiLU()
        
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        
    def forward(self, x: torch.Tensor, error: torch.Tensor = None,
                mode: str = "vanilla"):
        """
        Args:
            x: (B, L, D) input sequence
            error: (B, L, D) prediction error from PCN top-down (optional)
            mode: "vanilla" | "concat" | "delta" | "B" | "C"
        Returns:
            y: (B, L, D) output
        """
        B, L, D = x.shape
        
        # --- Convolution (local context) ---
        u = rearrange(x, 'b l d -> b d l')
        u = self.conv1d(u)[:, :, :L]
        u = self.act(u)
        u = rearrange(u, 'b d l -> b l d')
        
        # --- Compute Δ ---
        dt_rank = self.W_dt(u)  # (B, L, dt_rank)
        
        if error is not None:
            if mode == "delta":
                dt_rank = dt_rank + self.W_e_dt(error)
            elif mode == "concat":
                # Concat baseline: add error as extra input to u
                # Need to project error from d_model to d_model first, then add
                u = u + self.W_e_dt(error)  # reuse as additive injection
            # mode "B" and "C" don't affect Δ
        
        dt = self.dt_proj(dt_rank)  # (B, L, D)
        dt = F.softplus(dt + 0.5)   # positive bias for stability
        
        # --- Compute B, C ---
        B_proj = self.W_B(u)  # (B, L, d_state)
        C_proj = self.W_C(u)  # (B, L, d_state)
        
        if error is not None and mode == "B":
            B_proj = B_proj + self.W_e_B(error)
        elif error is not None and mode == "C":
            C_proj = C_proj + self.W_e_C(error)
        
        # --- Discretize A ---
        A = -torch.exp(self.A_log.float())  # (D, d_state)
        
        # Expand B, C to per-channel: (B, L, D, d_state)
        # B is shared across channels but scaled by per-channel Δ
        B_bar = B_proj.unsqueeze(2) * dt.unsqueeze(-1)  # (B,L,1,N)*(B,L,D,1)→(B,L,D,N)
        C = C_proj.unsqueeze(2).expand(-1, -1, D, -1)   # (B, L, D, N)
        
        # A_bar (B, L, D, N): per-channel, per-position discretization
        A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        
        # --- Selective scan ---
        y = selective_scan(u, A_bar, B_bar, C)
        
        return y


def selective_scan(u, A_bar, B_bar, C):
    """
    Parallel associative scan for selective SSM.
    
    Args:
        u: (B, L, D) input
        A_bar: (B, L, D, d_state) discretized A
        B_bar: (B, L, D, d_state) discretized B * Δ
        C: (B, L, D, d_state) output projection
    Returns:
        y: (B, L, D)
    """
    B, L, D, N = A_bar.shape
    
    # Reshape for scan: (B, D, L, N)
    A_bar = rearrange(A_bar, 'b l d n -> b d l n')
    B_bar = rearrange(B_bar, 'b l d n -> b d l n')
    C = rearrange(C, 'b l d n -> b d l n')
    u = rearrange(u, 'b l d -> b d l 1')
    
    # Sequential scan (O(L), but simple and correct for prototype)
    # For production, use pscan / selective_scan_ref from mamba repo
    h = torch.zeros(B, D, N, device=u.device, dtype=u.dtype)
    outputs = []
    
    for t in range(L):
        h = A_bar[:, :, t] * h + B_bar[:, :, t] * u[:, :, t]
        y_t = (h * C[:, :, t]).sum(dim=-1)  # (B, D)
        outputs.append(y_t)
    
    y = torch.stack(outputs, dim=-1)  # (B, D, L)
    y = rearrange(y, 'b d l -> b l d')
    
    return y


class MambaBlock(nn.Module):
    """Full Mamba block with gate, residual, and layer norm."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # gate + value
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)
        
        # Project error from d_model to d_inner for SSM modulation
        self.error_proj = nn.Linear(d_model, d_inner, bias=False)
        
    def forward(self, x: torch.Tensor, error: torch.Tensor = None,
                mode: str = "vanilla"):
        residual = x
        x = self.norm(x)

        has_seq = (x.dim() == 3)  # (B, L, D) patch mode vs (B, D) legacy

        # Project and split gate
        proj = self.in_proj(x)              # (B, D) → (B, d_inner*2) or (B, L, D) → (B, L, d_inner*2)
        gate, value = proj.chunk(2, dim=-1)

        # Ensure sequence dim: (B, d_inner) → (B, 1, d_inner)
        if not has_seq:
            value = value.unsqueeze(1)
            gate = gate.unsqueeze(1)

        # Project error to inner dim
        if error is not None and mode != "vanilla":
            error = self.error_proj(error)
            if not has_seq:
                error = error.unsqueeze(1)

        # SSM
        value = self.ssm(value, error, mode)

        # Gate + remove seq dim if added
        if not has_seq:
            value = value.squeeze(1)
            gate = gate.squeeze(1)
        out = value * F.silu(gate)
        out = self.out_proj(out)

        return out + residual
