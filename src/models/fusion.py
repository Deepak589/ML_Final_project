from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODES = {"concat", "attention", "ingr_only"}


class FusionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        mode: str,
        n_heads: int = 8,
        text_hidden: int = 768,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {_MODES}")
        self.mode = mode
        if mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        elif mode == "attention":
            self.ingr_token_proj = nn.Linear(text_hidden, embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ingr_emb: torch.Tensor,
        instr_emb: torch.Tensor,
        ingr_hidden: torch.Tensor | None = None,
        ingr_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([ingr_emb, instr_emb], dim=-1)
            return F.normalize(self.mlp(x), dim=-1)
        if self.mode == "attention":
            kv = self.ingr_token_proj(ingr_hidden)           # (B, L, embed_dim)
            q = instr_emb.unsqueeze(1)                        # (B, 1, embed_dim)
            key_padding_mask = ingr_mask == 0                 # True = ignore
            attended, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
            out = self.norm(attended.squeeze(1) + instr_emb)
            return F.normalize(out, dim=-1)
        # ingr_only
        return F.normalize(ingr_emb, dim=-1)
