from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.weight is not None:
            sample_weights = self.weight.to(logits.device)[targets]
            loss = sample_weights * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(
        self,
        samples_per_class: list[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()

        counts = torch.tensor(samples_per_class, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)

        effective_num = 1.0 - torch.pow(
            torch.tensor(beta, dtype=torch.float32),
            counts,
        )

        weights = (1.0 - beta) / torch.clamp(effective_num, min=1e-12)
        weights = weights / weights.sum() * len(samples_per_class)

        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.register_buffer("weights", weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)

        sample_weights = self.weights.to(logits.device)[targets]
        loss = sample_weights * ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_loss(
    loss_name: str,
    class_weights: torch.Tensor | None = None,
    samples_per_class: list[int] | None = None,
    focal_gamma: float = 2.0,
    cb_beta: float = 0.999,
) -> nn.Module:
    loss_name = loss_name.lower().strip()

    if loss_name == "ce":
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_ce":
        if class_weights is None:
            raise ValueError("weighted_ce wymaga class_weights.")
        return nn.CrossEntropyLoss(weight=class_weights.float())

    if loss_name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            weight=class_weights.float() if class_weights is not None else None,
        )

    if loss_name == "cb_focal":
        if samples_per_class is None:
            raise ValueError("cb_focal wymaga samples_per_class.")
        return ClassBalancedFocalLoss(
            samples_per_class=samples_per_class,
            beta=cb_beta,
            gamma=focal_gamma,
        )

    raise ValueError(f"Nieznana funkcja straty: {loss_name}")