# Dataset Adaptation Design: Recipe1M → Kaggle Food Dataset

**Date:** 2026-06-18  
**Status:** Approved  
**Context:** Recipe1M registration unavailable; course project — paper comparability not required.

---

## Problem

Original plan assumed Recipe1M format (`layer1.json`, `layer2.json`, `det_ingrs.json`, canonical `partition` field). That dataset is inaccessible. Need alternative with paired (image, recipe) data.

## Chosen Approach

Use Kaggle dataset `pes12017000148/food-ingredients-and-recipe-dataset-with-images` (~13k recipes + images). Add a thin adapter layer that normalizes its format into the project's internal representation. Everything downstream (dataset.py, tokenizers, models, losses, train, eval) is unchanged.

---

## What Changes

| Component | Before | After |
|---|---|---|
| Data source | `layer1.json` + `layer2.json` + `det_ingrs.json` | Kaggle CSV/JSON (single file) |
| Split generation | `partition` field from Recipe1M | Custom 80/10/10 random, seed=42 |
| Real-run scale | 50k recipes | ~10k (all available) |
| Eval subsets | 1k + 10k | 1k only |
| New file | — | `src/data/kaggle_adapter.py` |
| All other files | — | Unchanged |

---

## Internal Normalized Format

`kaggle_adapter.load()` returns `List[dict]`. Every dict:

```python
{
  "id": str,             # unique recipe identifier
  "title": str,
  "ingredients": List[str],  # one string per ingredient
  "instructions": str,       # full instruction text (joined if stored as list)
  "image_path": str,         # absolute path to image file
  "partition": str,          # "train" | "val" | "test"
}
```

`build_dataset.py` consumes this list. No Recipe1M-specific logic in build_dataset or anywhere downstream.

---

## Config Changes (`configs/data.yaml`)

**Remove:**
```yaml
paths.layer1
paths.layer2
paths.det_ingrs
splits.use_partition_field
```

**Add/Update:**
```yaml
paths:
  raw: data/raw
  processed: data/processed
  kaggle_data: data/raw/<filename-confirmed-at-download>
  images: data/raw/Food Images

subset:
  n_recipes: 2000      # smoke; bump to 10000 for real run
  require_image: true
  seed: 42

splits:
  train_frac: 0.80
  val_frac: 0.10
  test_frac: 0.10
```

`configs/baseline.yaml` and `configs/fusion.yaml`: no changes.

---

## Adapter Contract

`src/data/kaggle_adapter.py`:

```python
def load(cfg: DictConfig) -> List[dict]:
    """
    Read Kaggle dataset files and return normalized recipe dicts.
    Exact file format (CSV vs JSON, column names) resolved at download time.
    Only this function changes if source format differs from expectation.
    """
```

Split assignment (80/10/10, seed=42) happens inside `build_dataset.py` after `load()`, same as before.

---

## Phase / Todo Changes

| Phase | Change |
|---|---|
| Phase 1, step 7 | `build_dataset.py` calls `kaggle_adapter.load()` instead of reading layer1/layer2/det_ingrs |
| Phase 1, step 6 | Fixtures still 8 fake recipes — no change |
| Phase 2 | Precompute runs over `data/raw/Food Images/` on Kaggle notebook — same script |
| Phase 6 eval | Remove 10k subset; 1k eval only |
| Phase 7 baseline | `n_recipes` → 10k (not 50k) |

All other phases (models, losses, train loop, ablation, reproducibility) unchanged.

---

## Constraints / Risks

- **Exact Kaggle filenames unknown until download.** `kaggle_data` path in config is a placeholder; confirm and update after `kaggle datasets download`.
- **~13k total recipes.** After filtering to require_image, usable count may drop slightly. If <10k after filter, set `n_recipes` to actual count.
- **No `det_ingrs` file.** Ingredients come from recipe text. Ingredient tokenizer (`ingr_max_tokens: 24`) handles raw ingredient strings — no structural change needed.
- **1k eval only.** Not comparable to published im2recipe baselines. Acceptable for course demonstration.
