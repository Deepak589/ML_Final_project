# Phase 1: Kaggle Data Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Recipe1M data loading with a thin Kaggle adapter that normalizes ~13k food recipes into the project's internal dict format, update configs, and verify with fixture-based tests.

**Architecture:** A single `kaggle_adapter.load(cfg)` function reads the Kaggle CSV, assigns 80/10/10 splits (seed=42), and returns `List[dict]` matching the internal schema. Config is updated to remove Recipe1M paths. Downstream code is untouched.

**Tech Stack:** Python 3.11, OmegaConf, pandas (CSV parsing), pytest

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Modify | `configs/data.yaml` | Remove Recipe1M paths, add Kaggle paths |
| Modify | `configs/baseline.yaml` | Update `eval.subsets` to `[1k]` only |
| Create | `src/data/kaggle_adapter.py` | Load Kaggle CSV → `List[dict]` |
| Create | `tests/fixtures/mini_kaggle.csv` | 10-row fake CSV for unit tests |
| Create | `tests/test_phase1_kaggle_adapter.py` | Unit + integration tests |

---

### Task 1: Update `configs/data.yaml`

**Files:**
- Modify: `configs/data.yaml`

- [ ] **Step 1: Replace data.yaml content**

```yaml
# Data pipeline config. Single source of truth for paths + subset params.
paths:
  raw: data/raw
  processed: data/processed
  kaggle_csv: data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv
  images: data/raw/Food Images

subset:
  n_recipes: 2000               # smoke value; bump to 10000 for real run
  require_image: true           # keep only recipes with a non-empty image name
  seed: 42

splits:
  train_frac: 0.80
  val_frac: 0.10
  test_frac: 0.10

text:
  ingr_max_tokens: 24
  instr_max_tokens: 128
  tokenizer: distilbert-base-uncased

image:
  clip_model: ViT-B-32
  clip_pretrained: openai
  feat_dim: 512

semantic:
  enabled: true
  n_categories: 20
```

- [ ] **Step 2: Update eval.subsets in baseline.yaml**

In `configs/baseline.yaml`, change:
```yaml
eval:
  subsets: [1k, 10k]
```
to:
```yaml
eval:
  subsets: [1k]
```

- [ ] **Step 3: Verify configs still load**

```bash
cd /Users/deepakkatukuri/ML_final_project
python -c "from src.utils.config import load_config; cfg = load_config('baseline.yaml'); print(cfg.data.paths.kaggle_csv)"
```
Expected output: `data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv`

- [ ] **Step 4: Run existing tests to confirm no regression**

```bash
pytest tests/test_phase0_scaffold.py -v
```
Expected: 5 passed (note: `test_baseline_config_loads_with_data_include` checks `cfg.data.subset.n_recipes == 2000` — still true)

- [ ] **Step 5: Commit**

```bash
git add configs/data.yaml configs/baseline.yaml
git commit -m "config: adapt data.yaml for Kaggle dataset, eval subsets → [1k]"
```

---

### Task 2: Create test fixture CSV

**Files:**
- Create: `tests/fixtures/mini_kaggle.csv`

- [ ] **Step 1: Create the fixture CSV**

The real Kaggle dataset (`Food Ingredients and Recipe Dataset with Image Name Mapping.csv`) has these columns:
- `Title` — recipe name
- `Ingredients` — Python list literal as string, e.g. `"['flour', 'sugar', 'eggs']"`
- `Instructions` — plain text
- `Image_Name` — filename stem (no extension), e.g. `0080b432c30a0487`; empty string if no image

Create `tests/fixtures/mini_kaggle.csv` with 10 rows (8 with images, 2 without):

```csv
Title,Ingredients,Instructions,Image_Name
Spaghetti Bolognese,"['spaghetti', 'beef mince', 'tomato sauce']",Cook pasta. Brown beef. Mix.,img001
Chicken Curry,"['chicken', 'curry powder', 'coconut milk']",Fry chicken. Add spices. Simmer.,img002
Caesar Salad,"['romaine lettuce', 'parmesan', 'croutons']",Toss ingredients with dressing.,img003
Beef Tacos,"['beef', 'taco shells', 'salsa']",Season beef. Fill shells.,img004
Mushroom Risotto,"['arborio rice', 'mushrooms', 'parmesan']",Toast rice. Add stock gradually.,img005
Pancakes,"['flour', 'eggs', 'milk', 'butter']",Mix batter. Fry in pan.,img006
Greek Salad,"['cucumber', 'tomatoes', 'feta', 'olives']",Chop and combine.,img007
Lemon Chicken,"['chicken', 'lemon', 'garlic', 'herbs']",Marinate. Roast at 200C.,img008
Mystery Dish,"['unknown ingredient']",No instructions.,
Another No Image,"['butter', 'sugar']",Just mix.,
```

- [ ] **Step 2: Verify CSV is valid**

```bash
python -c "import csv; rows = list(csv.DictReader(open('tests/fixtures/mini_kaggle.csv'))); print(len(rows), 'rows'); print(rows[0].keys())"
```
Expected: `10 rows` and `dict_keys(['Title', 'Ingredients', 'Instructions', 'Image_Name'])`

---

### Task 3: Write failing tests for kaggle_adapter

**Files:**
- Create: `tests/test_phase1_kaggle_adapter.py`

- [ ] **Step 1: Write the test file**

```python
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
    """80/10/10 split on 8 recipes: expect ~6 train, ~1 val, ~1 test (may be 6/1/1)."""
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
```

- [ ] **Step 2: Run tests to confirm they all fail (ImportError expected)**

```bash
pytest tests/test_phase1_kaggle_adapter.py -v 2>&1 | head -30
```
Expected: all fail with `ModuleNotFoundError: No module named 'src.data.kaggle_adapter'`

---

### Task 4: Implement `src/data/kaggle_adapter.py`

**Files:**
- Create: `src/data/kaggle_adapter.py`

- [ ] **Step 1: Verify pandas is available**

```bash
python -c "import pandas; print(pandas.__version__)"
```
If missing: `pip install pandas`

- [ ] **Step 2: Write the implementation**

```python
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
```

- [ ] **Step 3: Run all phase 1 tests**

```bash
pytest tests/test_phase1_kaggle_adapter.py -v
```
Expected: all 11 tests pass

- [ ] **Step 4: Run full test suite (no regressions)**

```bash
pytest -v
```
Expected: all 16 tests pass (5 phase0 + 11 phase1)

- [ ] **Step 5: Commit**

```bash
git add src/data/kaggle_adapter.py tests/test_phase1_kaggle_adapter.py tests/fixtures/mini_kaggle.csv
git commit -m "feat(data): add kaggle_adapter with 80/10/10 split and fixture tests"
```

---

### Task 5: Download note + path verification

> **NOTE:** This task requires Kaggle dataset to be downloaded. Skip if running in CI without data.

- [ ] **Step 1: Download dataset (run in terminal)**

```bash
cd /Users/deepakkatukuri/ML_final_project
mkdir -p data/raw
kaggle datasets download pes12017000148/food-ingredients-and-recipe-dataset-with-images -p data/raw --unzip
```

- [ ] **Step 2: Confirm exact filename**

```bash
ls "data/raw/" | grep -i food
```
Expected output should include the CSV filename. If different from `Food Ingredients and Recipe Dataset with Image Name Mapping.csv`, update `configs/data.yaml` → `paths.kaggle_csv`.

- [ ] **Step 3: Confirm images directory name**

```bash
ls "data/raw/" | grep -i image
```
Expected: `Food Images` (or similar). Update `configs/data.yaml` → `paths.images` if different.

- [ ] **Step 4: Smoke test adapter against real data**

```python
# run from project root
python - <<'EOF'
from src.utils.config import load_config
from src.data.kaggle_adapter import load

cfg = load_config("baseline.yaml")
recipes = load(cfg.data)
print(f"Loaded {len(recipes)} recipes")
print(f"Example: {recipes[0]}")
splits = {p: sum(1 for r in recipes if r['partition'] == p) for p in ['train', 'val', 'test']}
print(f"Split counts: {splits}")
EOF
```
Expected: loads ~2000 recipes (n_recipes=2000 smoke value), prints valid dict, shows ~1600/200/200 split.

- [ ] **Step 5: Commit config if paths needed adjustment**

```bash
git add configs/data.yaml
git commit -m "config: confirm Kaggle dataset paths after download"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: config update ✓, kaggle_adapter.py ✓, split 80/10/10 ✓, require_image filter ✓, n_recipes cap ✓, internal dict format ✓, eval subsets→[1k] ✓
- [x] **No placeholders**: all steps have exact code/commands
- [x] **Type consistency**: `load()` signature consistent across tests and impl; `_parse_ingredients` / `_parse_instructions` / `_assign_splits` referenced only in impl file
- [x] **Downstream unchanged**: adapter returns `List[dict]` matching spec; no changes to models/losses/train/eval
