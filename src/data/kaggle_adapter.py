"""Kaggle food-recipe dataset adapter.

Reads the Kaggle CSV (pes12017000148/food-ingredients-and-recipe-dataset-with-images),
normalizes to the internal recipe dict format, and assigns 80/10/10 splits.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


def load(cfg: DictConfig) -> list[dict]:
    """Load Kaggle dataset and return normalized recipe dicts.

    Args:
        cfg: OmegaConf config with keys:
            cfg.paths.kaggle_csv  — path to the CSV file
            cfg.paths.images      — path to images directory
            cfg.subset.n_recipes  — max recipes to keep (applied after filtering)
            cfg.subset.require_image — drop rows with no Image_Name
            cfg.subset.seed       — RNG seed for shuffling before cap
            cfg.splits.train_frac / val_frac / test_frac

    Returns:
        List of dicts with keys: id, title, ingredients, instructions, image_path, partition
    """
    csv_path = Path(cfg.paths.kaggle_csv)
    images_dir = Path(cfg.paths.images)

    df = pd.read_csv(csv_path)

    # normalize column names: strip whitespace
    df.columns = df.columns.str.strip()

    if cfg.subset.require_image:
        df = df[df["Image_Name"].notna() & (df["Image_Name"].str.strip() != "")]

    # shuffle deterministically, then cap
    rng = np.random.default_rng(cfg.subset.seed)
    idx = rng.permutation(len(df))
    df = df.iloc[idx].reset_index(drop=True)
    df = df.iloc[: cfg.subset.n_recipes]

    recipes = []
    for row_idx, row in df.iterrows():
        recipe_id = str(row_idx)
        title = str(row["Title"]).strip()
        ingredients = _parse_ingredients(row["Ingredients"])
        instructions = _parse_instructions(row["Instructions"])
        image_name = str(row["Image_Name"]).strip()
        image_path = str(images_dir / f"{image_name}.jpg")
        recipes.append({
            "id": recipe_id,
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions,
            "image_path": image_path,
            "partition": "",  # filled in next step
        })

    _assign_splits(recipes, cfg.splits, cfg.subset.seed)
    return recipes


def _parse_ingredients(raw) -> list[str]:
    """Parse ingredient field: Python list literal string → List[str]."""
    if isinstance(raw, list):
        return [str(i).strip() for i in raw if str(i).strip()]
    raw = str(raw).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(i).strip() for i in parsed if str(i).strip()]
    except (ValueError, SyntaxError):
        pass
    # fallback: comma-split
    return [s.strip() for s in raw.split(",") if s.strip()]


def _parse_instructions(raw) -> str:
    """Parse instruction field: list literal or plain string → single string."""
    if isinstance(raw, list):
        return " ".join(str(s).strip() for s in raw)
    raw = str(raw).strip()
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return " ".join(str(s).strip() for s in parsed)
    except (ValueError, SyntaxError):
        pass
    return raw


def _assign_splits(recipes: list[dict], splits_cfg: DictConfig, seed: int) -> None:
    """Assign 'partition' in-place using deterministic index-based split."""
    n = len(recipes)
    n_train = round(n * splits_cfg.train_frac)
    n_val = round(n * splits_cfg.val_frac)

    for i, recipe in enumerate(recipes):
        if i < n_train:
            recipe["partition"] = "train"
        elif i < n_train + n_val:
            recipe["partition"] = "val"
        else:
            recipe["partition"] = "test"
