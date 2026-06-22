from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, image_emb: torch.Tensor, recipe_emb: torch.Tensor
    ) -> torch.Tensor:
        B = image_emb.size(0)
        labels = torch.arange(B, device=image_emb.device)
        logits_i2r = image_emb @ recipe_emb.T / self.temperature
        logits_r2i = recipe_emb @ image_emb.T / self.temperature
        return 0.5 * (
            F.cross_entropy(logits_i2r, labels) + F.cross_entropy(logits_r2i, labels)
        )
