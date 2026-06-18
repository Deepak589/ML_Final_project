"""Phase 1: kaggle_adapter.load() contract tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "mini_kaggle.csv"
FIXTURE_IMAGES_DIR = Path(__file__).parent / "fixtures" / "food_images"


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "paths": {
            "kaggle_csv": str(FIXTURE_CSV),
            "images": str(FIXTURE_IMAGES_DIR),
        },
        "subset": {
            "n_recipes": 100,      # larger than fixture — return all
            "require_image": True,
            "seed": 42,
        },
        "splits": {
            "train_frac": 0.80,
            "val_frac": 0.10,
            "test_frac": 0.10,
        },
    })


def test_load_returns_list_of_dicts(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    assert isinstance(recipes, list)
    assert len(recipes) > 0
    assert isinstance(recipes[0], dict)


def test_required_keys_present(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    required = {"id", "title", "ingredients", "instructions", "image_path", "partition"}
    for r in recipes:
        assert required == set(r.keys()), f"Missing keys: {required - set(r.keys())}"


def test_ingredients_is_list_of_strings(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    for r in recipes:
        assert isinstance(r["ingredients"], list)
        assert all(isinstance(i, str) for i in r["ingredients"])


def test_partition_values_valid(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    valid = {"train", "val", "test"}
    for r in recipes:
        assert r["partition"] in valid


def test_require_image_filters_blank_image_name(cfg):
    """Rows with empty Image_Name must be dropped when require_image=True."""
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    # fixture has 8 rows with images, 2 without
    assert len(recipes) == 8


def test_require_image_false_keeps_all(cfg):
    from src.data.kaggle_adapter import load
    cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"subset": {"require_image": False}}))
    recipes = load(cfg2)
    assert len(recipes) == 10


def test_split_fractions_approximate(cfg):
    """80/10/10 split on 8 recipes: expect ~6 train, ~1 val, ~1 test."""
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    counts = {"train": 0, "val": 0, "test": 0}
    for r in recipes:
        counts[r["partition"]] += 1
    assert counts["train"] >= 5
    assert counts["val"] >= 1
    assert counts["test"] >= 1
    assert sum(counts.values()) == 8


def test_split_reproducible_with_seed(cfg):
    from src.data.kaggle_adapter import load
    r1 = [r["partition"] for r in load(cfg)]
    r2 = [r["partition"] for r in load(cfg)]
    assert r1 == r2


def test_n_recipes_cap(cfg):
    from src.data.kaggle_adapter import load
    cfg2 = OmegaConf.merge(cfg, OmegaConf.create({"subset": {"n_recipes": 4, "require_image": False}}))
    recipes = load(cfg2)
    assert len(recipes) == 4


def test_image_path_is_string(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    for r in recipes:
        assert isinstance(r["image_path"], str)
        assert len(r["image_path"]) > 0


def test_instructions_is_string(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    for r in recipes:
        assert isinstance(r["instructions"], str)
        assert len(r["instructions"]) > 0


def test_id_unique(cfg):
    from src.data.kaggle_adapter import load
    recipes = load(cfg)
    ids = [r["id"] for r in recipes]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"
