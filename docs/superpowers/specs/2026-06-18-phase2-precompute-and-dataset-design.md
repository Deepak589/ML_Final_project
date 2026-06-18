# Phase 2: Precompute + Dataset Design Spec

**Date:** 2026-06-18
**Status:** Approved
**Context:** Phase 1 complete — `kaggle_adapter.load()` returns normalized `List[dict]`. Phase 2 builds the precomputation pipeline and PyTorch Dataset on top.

---

## Goal

1. Fix nested image directory path in config.
2. Script to precompute CLIP ViT-B-32 image features → cached `.pt` file.
3. `RecipeDataset` — PyTorch Dataset that loads recipes + cached image features, tokenizes text on-the-fly, returns per-sample dicts for the training loop.

Everything downstream (models, losses, train loop) is unchanged.

---

## What Changes

| Component | Action |
|---|---|
| `configs/data.yaml` | Fix `paths.images` → `data/raw/Food Images/Food Images` |
| `src/data/precompute_image_feats.py` | New script |
| `src/data/build_dataset.py` | New module |
| `tests/test_phase2_precompute.py` | New test file |
| `tests/test_phase2_dataset.py` | New test file |
| `tests/fixtures/` | Add fake `image_feats.pt` + small JPEGs |

---

## Config Fix

`configs/data.yaml` `paths.images` must be:

```yaml
images: data/raw/Food Images/Food Images
```

The downloaded Kaggle zip nests images one level deeper than the top-level folder name.

Also add the processed features path:

```yaml
paths:
  ...
  image_feats: data/processed/image_feats.pt
```

---

## Precompute Script (`src/data/precompute_image_feats.py`)

### Contract

```python
def main(cfg: DictConfig) -> None:
    """
    Load all images from cfg.data.paths.images, run CLIP ViT-B-32,
    save {image_stem: Tensor(512)} to cfg.data.paths.image_feats.
    """
```

### Behavior

- Reads all `.jpg` files from `cfg.data.paths.images`
- Runs CLIP ViT-B-32 (`open_clip`, `pretrained="openai"`) with `torch.no_grad()`
- Processes in batches of 64 (configurable via `--batch-size` CLI arg)
- Device selected via `resolve_device(cfg.device)`
- Saves output: `torch.save(feats_dict, cfg.data.paths.image_feats)`
  - `feats_dict`: `{stem: Tensor(512)}` where `stem = path.stem` (filename without `.jpg`)
- Creates `data/processed/` if it doesn't exist
- Logs: total images found, batch progress (tqdm), images skipped (load errors), final count saved
- Idempotent: overwrites existing file

### CLI Usage

```bash
python -m src.data.precompute_image_feats --config baseline.yaml
python -m src.data.precompute_image_feats --config baseline.yaml --batch-size 128 --device cuda
```

### Error handling

- Image load failure (corrupt file): log warning, skip, continue
- `image_feats` parent dir missing: create it
- No images found: raise `FileNotFoundError` with helpful message

---

## Dataset (`src/data/build_dataset.py`)

### Class

```python
class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, partition: str | None = None): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict: ...
```

### `__init__` behavior

1. Call `kaggle_adapter.load(cfg.data)` → `List[dict]`
2. Filter by `partition` if provided (`"train"`, `"val"`, `"test"`, or `None` = all)
3. Load `torch.load(cfg.data.paths.image_feats)` → `feats_dict`
4. Filter recipes to those present in `feats_dict` (log count of dropped recipes)
5. Initialize `AutoTokenizer.from_pretrained(cfg.data.text.tokenizer)`
6. Store `self.recipes`, `self.feats_dict`, `self.tokenizer`, `self.cfg`

### `__getitem__` return dict

```python
{
    "image_feat":          Tensor(512),   # float32, from feats_dict
    "ingr_input_ids":      Tensor(ingr_max_tokens),
    "ingr_attention_mask": Tensor(ingr_max_tokens),
    "instr_input_ids":     Tensor(instr_max_tokens),
    "instr_attention_mask":Tensor(instr_max_tokens),
    "recipe_id":           str,
    "partition":           str,
}
```

### Text tokenization

- **Ingredients**: join `recipe["ingredients"]` with `", "` separator → tokenize with `max_length=cfg.data.text.ingr_max_tokens`, `truncation=True`, `padding="max_length"`
- **Instructions**: `recipe["instructions"]` → tokenize with `max_length=cfg.data.text.instr_max_tokens`, same settings
- Use `Cleaned_Ingredients` from CSV if present; `kaggle_adapter` already parses this via the `ingredients` field

### Helper

```python
def get_split(cfg: DictConfig, partition: str) -> RecipeDataset:
    """Convenience wrapper: RecipeDataset(cfg, partition=partition)"""
```

---

## Data Flow

```
configs/data.yaml
       ↓
kaggle_adapter.load(cfg.data)          → List[dict] (id, title, ingredients,
                                                      instructions, image_path,
                                                      partition)
       ↓
RecipeDataset.__init__(cfg, partition)
  - filters by partition
  - loads image_feats.pt → {stem: Tensor(512)}
  - inits tokenizer
       ↓
RecipeDataset.__getitem__(i)
  - tokenizes ingr + instr (on-the-fly)
  - looks up image_feat by stem from image_path
  - returns sample dict
       ↓
torch.utils.data.DataLoader
  → batches to training loop (Phase 3+)
```

---

## Image Stem Lookup

`image_path` in each recipe is `str(images_dir / f"{Image_Name}.jpg")`.
To look up in `feats_dict`: `stem = Path(recipe["image_path"]).stem`.
This matches the key written by the precompute script (`path.stem`).

---

## Tests

### `tests/test_phase2_precompute.py`

Uses a `tmp_path` pytest fixture with 3 small synthetic JPEGs (1×1 pixel, PIL-generated). Runs precompute logic (mocked or real CLIP on CPU for speed). Verifies:
- Output file created at correct path
- Dict has correct keys (stems)
- Each value is `Tensor` of shape `(512,)`
- Corrupt image skipped gracefully

### `tests/test_phase2_dataset.py`

Uses `tests/fixtures/mini_kaggle.csv` + synthetic `image_feats.pt` (random tensors for img001–img008). Verifies:
- `len(dataset)` matches filtered recipe count
- `__getitem__` returns dict with all required keys + correct types/shapes
- `get_split("train")` returns only train-partition recipes
- Recipe missing from `feats_dict` is dropped (not raises)
- Tokenization produces tensors of correct max-length shape

---

## Constraints

- CLIP via `open_clip` (`pip install open-clip-torch`) — already in model config (`clip_model: ViT-B-32`, `clip_pretrained: openai`)
- DistilBERT tokenizer via `transformers` (`AutoTokenizer`) — already in config
- `tqdm` for progress — add to `requirements.txt` if not present
- `data/processed/` is gitignored (generated artifacts)
- Precompute script is run once; Dataset assumes `image_feats.pt` exists (raises `FileNotFoundError` with clear message if missing)
