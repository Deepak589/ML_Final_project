"""Phase 2: RecipeDataset contract tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "mini_kaggle.csv"
FIXTURE_IMAGES_DIR = Path(__file__).parent / "fixtures" / "food_images"


@pytest.fixture
def image_feats_path(tmp_path):
    feats = {f"img00{i}": torch.randn(512) for i in range(1, 9)}
    pt_path = tmp_path / "image_feats.pt"
    torch.save(feats, pt_path)
    return pt_path


@pytest.fixture
def cfg(image_feats_path):
    return OmegaConf.create({
        "data": {
            "paths": {
                "kaggle_csv": str(FIXTURE_CSV),
                "images": str(FIXTURE_IMAGES_DIR),
                "image_feats": str(image_feats_path),
            },
            "text": {
                "tokenizer": "distilbert-base-uncased",
                "ingr_max_tokens": 24,
                "instr_max_tokens": 128,
            },
            "image": {"clip_model": "ViT-B-32", "clip_pretrained": "openai", "feat_dim": 512},
            "subset": {"n_recipes": 100, "require_image": True, "seed": 42},
            "splits": {"train_frac": 0.80, "val_frac": 0.10, "test_frac": 0.10},
        },
    })


def test_len_all_recipes_with_feats(cfg):
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(cfg)
    assert len(ds) == 8  # all 8 image-named fixture rows have feats


def test_getitem_keys(cfg):
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(cfg)
    sample = ds[0]
    assert set(sample.keys()) == {
        "image_feat",
        "ingr_input_ids",
        "ingr_attention_mask",
        "instr_input_ids",
        "instr_attention_mask",
        "recipe_id",
        "partition",
    }


def test_getitem_image_feat_shape(cfg):
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(cfg)
    sample = ds[0]
    assert isinstance(sample["image_feat"], torch.Tensor)
    assert sample["image_feat"].shape == (512,)
    assert sample["image_feat"].dtype == torch.float32


def test_getitem_token_shapes(cfg):
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(cfg)
    sample = ds[0]
    assert sample["ingr_input_ids"].shape == (24,)
    assert sample["ingr_attention_mask"].shape == (24,)
    assert sample["instr_input_ids"].shape == (128,)
    assert sample["instr_attention_mask"].shape == (128,)


def test_getitem_metadata_types(cfg):
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(cfg)
    sample = ds[0]
    assert isinstance(sample["recipe_id"], str)
    assert isinstance(sample["partition"], str)
    assert sample["partition"] in {"train", "val", "test"}


def test_get_split_train_only(cfg):
    from src.data.build_dataset import get_split
    ds = get_split(cfg, "train")
    assert all(r["partition"] == "train" for r in ds.recipes)
    assert len(ds) > 0


def test_missing_feat_recipe_dropped(cfg, tmp_path):
    # feats only for img001–img004; img005–img008 missing
    partial_feats = {f"img00{i}": torch.randn(512) for i in range(1, 5)}
    pt_path = tmp_path / "partial.pt"
    torch.save(partial_feats, pt_path)

    partial_cfg = OmegaConf.merge(
        cfg, OmegaConf.create({"data": {"paths": {"image_feats": str(pt_path)}}})
    )
    from src.data.build_dataset import RecipeDataset
    ds = RecipeDataset(partial_cfg)
    assert len(ds) == 4  # only 4 recipes have feats


def test_partition_filter_none_returns_all(cfg):
    from src.data.build_dataset import RecipeDataset
    ds_all = RecipeDataset(cfg, partition=None)
    ds_train = RecipeDataset(cfg, partition="train")
    ds_val = RecipeDataset(cfg, partition="val")
    ds_test = RecipeDataset(cfg, partition="test")
    assert len(ds_train) + len(ds_val) + len(ds_test) == len(ds_all)
