"""RecipeDataset: loads cached image features + tokenizes text on-the-fly."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data import kaggle_adapter

_log = logging.getLogger(__name__)


class RecipeDataset(Dataset):
    def __init__(self, cfg: DictConfig, partition: str | None = None) -> None:
        recipes = kaggle_adapter.load(cfg.data)
        if partition is not None:
            recipes = [r for r in recipes if r["partition"] == partition]

        feat_path = Path(cfg.data.paths.image_feats)
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Image features not found at {feat_path}. "
                "Run: python -m src.data.precompute_image_feats --config baseline.yaml"
            )
        feats_dict: dict[str, torch.Tensor] = torch.load(
            feat_path, weights_only=False
        )

        before = len(recipes)
        recipes = [r for r in recipes if Path(r["image_path"]).stem in feats_dict]
        dropped = before - len(recipes)
        if dropped:
            _log.warning("Dropped %d recipes missing from image_feats", dropped)

        self.recipes = recipes
        self.feats_dict = feats_dict
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.data.text.tokenizer)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> dict:
        recipe = self.recipes[idx]
        stem = Path(recipe["image_path"]).stem
        image_feat = self.feats_dict[stem].float()

        ingr_enc = self.tokenizer(
            ", ".join(recipe["ingredients"]),
            max_length=self.cfg.data.text.ingr_max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        instr_enc = self.tokenizer(
            recipe["instructions"],
            max_length=self.cfg.data.text.instr_max_tokens,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "image_feat": image_feat,
            "ingr_input_ids": ingr_enc["input_ids"].squeeze(0),
            "ingr_attention_mask": ingr_enc["attention_mask"].squeeze(0),
            "instr_input_ids": instr_enc["input_ids"].squeeze(0),
            "instr_attention_mask": instr_enc["attention_mask"].squeeze(0),
            "recipe_id": recipe["id"],
            "partition": recipe["partition"],
        }


def get_split(cfg: DictConfig, partition: str) -> RecipeDataset:
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Unknown partition {partition!r}; expected train/val/test")
    return RecipeDataset(cfg, partition=partition)
