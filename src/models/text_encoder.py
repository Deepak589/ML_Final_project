from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden: int,
        embed_dim: int,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)
        self.ingr_proj = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.instr_proj = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim)
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

    def encode_ingr(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean(out.last_hidden_state, attention_mask)
        emb = F.normalize(self.ingr_proj(pooled), dim=-1)
        return emb, out.last_hidden_state

    def encode_instr(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean(out.last_hidden_state, attention_mask)
        return F.normalize(self.instr_proj(pooled), dim=-1)
