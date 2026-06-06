"""
PCN Layer with Mamba SSM backbone — simplified for Phase 1a prototype.
All representations are (B, D); SSM internally treats this as L=1 sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssm import MambaBlock


class PCNLayer(nn.Module):
    """
    One layer of a predictive coding network.
    
    Feedforward:  z^ℓ = Mamba(z^(ℓ-1))  [with optional error modulation]
    Top-down:     μ^ℓ = Decoder(z^(ℓ+1))
    Error:        e^ℓ = z^ℓ - μ^ℓ  (modulates SSM in next iteration)
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        
        self.mamba = MambaBlock(d_model, d_state)
        
        self.topdown = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
    def forward(self, z_below: torch.Tensor, z_above: torch.Tensor = None,
                mode: str = "vanilla", n_iter: int = 10):
        """
        Args:
            z_below: (B, D) representation from layer below
            z_above: (B, D) representation from layer above (top-down) or None
            mode: SSM ablation mode
            n_iter: number of PCN inference iterations
        Returns:
            z: (B, D) converged representation
            e: (B, D) final prediction error
        """
        B, D = z_below.shape
        
        # Initial feedforward (no error modulation)
        z = self.mamba(z_below, error=None, mode="vanilla")
        error = torch.zeros_like(z)
        
        for _ in range(n_iter):
            # Top-down prediction
            if z_above is not None:
                mu = self.topdown(z_above.detach())
                error = z - mu
            
            # Feedforward with error modulation (error=None for vanilla)
            effective_error = error if mode != "vanilla" else None
            z_new = self.mamba(z_below, error=effective_error, mode=mode)
            
            if torch.mean((z_new - z) ** 2) < 1e-5:
                break
            z = z_new
        
        return z, error


class PCNClassifier(nn.Module):
    """
    Full PC-S³ model.
    
    Architecture: 
        Input → [Linear proj] → [N × PCN Layer] → Classifier head
    """
    def __init__(self, input_dim: int, num_classes: int = 100,
                 d_model: int = 256, num_layers: int = 3, d_state: int = 16,
                 mode: str = "vanilla", n_iter: int = 10):
        super().__init__()
        
        self.mode = mode
        self.n_iter = n_iter
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            PCNLayer(d_model, d_state) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        
    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.flatten(1)
        
        x = self.input_proj(x)  # (B, d_model)
        
        # Bottom-up: single forward pass per layer
        z_list = [x]
        for layer in self.layers:
            z, _ = layer(z_list[-1], mode="vanilla", n_iter=1)
            z_list.append(z)
        
        # Top-down: PCN inference with error modulation
        z_top = None
        for i in range(len(self.layers) - 1, -1, -1):
            z, error = self.layers[i](
                z_list[i], z_above=z_top,
                mode=self.mode, n_iter=self.n_iter,
            )
            z_top = z
        
        return self.classifier(z_top)


def make_model(mode: str, input_dim: int = 3072, num_classes: int = 100,
               d_model: int = 256, num_layers: int = 3, n_iter: int = 10):
    return PCNClassifier(
        input_dim=input_dim, num_classes=num_classes,
        d_model=d_model, num_layers=num_layers,
        mode=mode, n_iter=n_iter,
    )
