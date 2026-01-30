from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


def _reduce(loss: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Invalid reduction: {reduction}")


@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    alpha: Optional[float] = 0.25
    reduction: Reduction = "mean"
    eps: float = 1e-6


class BinaryFocalLossWithLogits(nn.Module):
    """
    Binary focal loss for logits.

    Args:
        gamma: focusing parameter.
        alpha: class balancing factor for positive class.
               If None, no alpha balancing is applied.
    """

    def __init__(self, gamma: float = 1.2, alpha: Optional[float] = 0.8, reduction: Reduction = "sum"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        target = target.to(dtype=logit.dtype) # target : GT labels

        # BCE per element
        bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")

        # p_t = p if y=1 else (1-p)
        p = torch.sigmoid(logit)
        p_t = p * target + (1.0 - p) * (1.0 - target)

        # focal modulation
        mod = (1.0 - p_t).pow(self.gamma)

        # alpha balancing
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_t * mod * bce
        else:
            loss = mod * bce

        return _reduce(loss, self.reduction)


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for probability inputs.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = 0.25, reduction: Reduction = "mean", eps: float = 1e-6):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=prob.dtype)
        prob = prob.clamp(self.eps, 1.0 - self.eps)

        # p_t
        p_t = prob * target + (1.0 - prob) * (1.0 - target)

        # BCE from probabilities
        bce = -(target * torch.log(prob) + (1.0 - target) * torch.log(1.0 - prob))

        mod = (1.0 - p_t).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
            loss = alpha_t * mod * bce
        else:
            loss = mod * bce

        return _reduce(loss, self.reduction)
