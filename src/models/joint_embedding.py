from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import FusionModule


class JointEmbeddingModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        m = cfg.model
        self.image_encoder = ImageEncoder(
            in_dim=m.image.in_dim,
            proj_hidden=m.image.proj_hidden,
            embed_dim=m.embed_dim,
        )
        self.text_encoder = TextEncoder(
            encoder_name=m.text.encoder,
            hidden=m.text.hidden,
            embed_dim=m.embed_dim,
            freeze=m.text.freeze_text,
        )
        self.fusion = FusionModule(
            embed_dim=m.embed_dim,
            hidden=m.fusion.hidden,
            mode=m.fusion.mode,
            n_heads=m.fusion.get("n_heads", 8),
            text_hidden=m.text.hidden,
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image_emb = self.image_encoder(batch["image_feat"])
        ingr_emb, ingr_hidden = self.text_encoder.encode_ingr(
            batch["ingr_input_ids"], batch["ingr_attention_mask"]
        )
        instr_emb = self.text_encoder.encode_instr(
            batch["instr_input_ids"], batch["instr_attention_mask"]
        )
        recipe_emb = self.fusion(
            ingr_emb,
            instr_emb,
            ingr_hidden=ingr_hidden,
            ingr_mask=batch["ingr_attention_mask"],
        )
        return image_emb, recipe_emb
