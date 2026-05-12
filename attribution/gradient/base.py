from __future__ import annotations

"""
GradientAttributionBase
Shared helpers for gradient-based dataset attribution strategies.
"""

from typing import Callable, Mapping, Tuple, List

import torch
from torch import nn

from attribution.base import DataAttribution

# ------------------------- Type aliases -------------------------
Batch = Mapping[str, torch.Tensor]
LossFn = Callable[[nn.Module, Batch], torch.Tensor]


class GradientAttributionBase(DataAttribution):
    """Base class with reusable gradient helpers for gradient-based strategies.

    Provides:
    - trainable parameter selection helper
    - gradient extraction as state_dict
    - gradient zeroing
    - cosine & dot similarity utilities for state_dicts
    """

    # ---- Parameter utilities ----
    def _named_trainable_params(self, model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
        params: List[Tuple[str, nn.Parameter]] = []
        for name, p in model.named_parameters():
            if p.requires_grad and (self.trainable_param_filter(name, p) if self.trainable_param_filter else True):
                params.append((name, p))
        if not params:
            raise ValueError("No trainable parameters selected. Check trainable_param_filter or model setup.")
        return params

    def _grads_to_state_dict(self, model: nn.Module) -> dict:
        """Extract gradients as a state_dict (preserves parameter names and shapes)."""
        grad_dict = {}
        for name, p in self._named_trainable_params(model):
            if p.grad is None:
                grad_dict[name] = torch.zeros_like(p).detach().cpu()
            else:
                grad_dict[name] = p.grad.detach().cpu().clone()
        return grad_dict

    def _zero_grad(self, model: nn.Module):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # ---- Similarity ----
    def _cosine(self, a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        a_n = a / (a.norm(p=2) + eps)
        b_n = b / (b.norm(p=2) + eps)
        return (a_n * b_n).sum()

    def similarity(self, a: torch.Tensor, b: torch.Tensor, kind: str = "cos") -> float:
        a = a.to(self.device)
        b = b.to(self.device)
        if kind == "cos":
            return float(self._cosine(a, b).item())
        elif kind == "dot":
            return float((a * b).sum().item())
        else:
            raise ValueError("similarity must be 'cos' or 'dot'")

    def similarity_state_dict(self, a: dict, b: dict, kind: str = "cos") -> float:
        """Compute similarity between two gradient state_dicts."""
        # Flatten both state_dicts on CPU first
        a_flat = torch.cat([a[k].reshape(-1) for k in sorted(a.keys())], dim=0)
        b_flat = torch.cat([b[k].reshape(-1) for k in sorted(b.keys())], dim=0)
        
        # Move to GPU, compute similarity, then immediately free
        a_flat = a_flat.to(self.device)
        b_flat = b_flat.to(self.device)
        result = self.similarity(a_flat, b_flat, kind=kind)
        
        # Free GPU memory
        del a_flat, b_flat
        torch.cuda.empty_cache()
        
        return result