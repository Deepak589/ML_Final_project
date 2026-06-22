# Phase 2: Precompute + Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix image path config, precompute CLIP ViT-B-32 image features to disk, and implement `RecipeDataset` that loads cached features + tokenizes text on-the-fly.

**Architecture:** Thin precompute script runs once and saves `{stem: Tensor(512)}` to `data/processed/image_feats.pt`. `RecipeDataset` loads that cache at init, filters to recipes with matching stems, and tokenizes ingredients + instructions in `__getitem__`. No model runs during training data loading.

**Tech Stack:** `open_clip`, `transformers.AutoTokenizer`, `torch.utils.data.Dataset`, `omegaconf`, `tqdm`, `Pillow`

## Global Constraints

- Python 3.10+
- `src.utils.config.load_config(path)` handles `defaults: {data: data.yaml}` merge → results in `cfg.data.*` node
- `src.utils.config.resolve_device("auto")` → `"cuda" | "mps" | "cpu"`
- `kaggle_adapter.load(cfg.data)` takes `cfg.data` (not `cfg`)
- Fixture CSV at `tests/fixtures/mini_kaggle.csv` has 10 rows: img001–img008 (8 with images), 2 without
- All tests mock CLIP (avoid downloading ViT-B-32 weights); tokenizer uses real distilbert-base-uncased
- `data/processed/` is gitignored (generated artifacts)

---

### Task 1: Config Fix

**Files:**
- Modify: `configs/data.yaml`

**Interfaces:**
- Produces: `cfg.data.paths.images = "data/raw/Food Images/Food Images"`, `cfg.data.paths.image_feats = "data/processed/image_feats.pt"`

- [ ] **Step 1: Fix `paths.images` and add `paths.image_feats` in `configs/data.yaml`**

Open `configs/data.yaml`. Replace the `paths:` block:

```yaml
paths:
  raw: data/raw
  processed: data/processed
  kaggle_csv: data/raw/Food Ingredients and Recipe Dataset with Image Name Mapping.csv
  images: data/raw/Food Images/Food Images
  image_feats: data/processed/image_feats.pt
```

The Kaggle zip nests images one extra level: `Food Images/Food Images/*.jpg`.

- [ ] **Step 2: Verify load_config picks up the new key**

```bash
python - <<'EOF'
from src.utils.config import load_config
cfg = load_config("baseline.yaml")
print(cfg.data.paths.images)
print(cfg.data.paths.image_feats)
EOF
```

Expected output:
```
data/raw/Food Images/Food Images
data/processed/image_feats.pt
```

- [ ] **Step 3: Commit**

```bash
git add configs/data.yaml
git commit -m "config: fix images nested path and add image_feats path"
```

---

### Task 2: Precompute Script (TDD)

**Files:**
- Create: `src/data/precompute_image_feats.py`
- Create: `tests/test_phase2_precompute.py`

**Interfaces:**
- Consumes: `cfg.data.paths.images`, `cfg.data.paths.image_feats`, `cfg.data.image.clip_model`, `cfg.data.image.clip_pretrained`, `cfg.device`, `cfg.batch_size`
- Consumes: `src.utils.config.load_config`, `src.utils.config.resolve_device`
- Produces: `main(cfg: DictConfig) -> None` — saves `{str: Tensor(512)}` dict to `cfg.data.paths.image_feats`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase2_precompute.py`:

```python
"""Phase 2: precompute_image_feats.main() contract tests. CLIP is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


@pytest.fixture
def tmp_images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        Image.new("RGB", (1, 1), color=(128, 64, 32)).save(img_dir / f"{name}.jpg")
    return img_dir


@pytest.fixture
def tmp_cfg(tmp_path, tmp_images):
    return OmegaConf.create({
        "data": {
            "paths": {
                "images": str(tmp_images),
                "image_feats": str(tmp_path / "processed" / "image_feats.pt"),
            },
            "image": {"clip_model": "ViT-B-32", "clip_pretrained": "openai"},
        },
        "device": "cpu",
        "batch_size": 64,
    })


def _make_mock_clip():
    model = MagicMock()
    model.encode_image.side_effect = lambda batch: torch.randn(batch.shape[0], 512)
    preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    return model, preprocess


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_output_file_created(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    assert Path(tmp_cfg.data.paths.image_feats).exists()


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_dict_keys_are_stems(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    assert set(feats.keys()) == {"alpha", "beta", "gamma"}


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_feat_shape_is_512(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    for stem, feat in feats.items():
        assert isinstance(feat, torch.Tensor), f"{stem}: expected Tensor"
        assert feat.shape == (512,), f"{stem}: expected (512,), got {feat.shape}"


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_corrupt_image_skipped(mock_create, tmp_cfg, tmp_images):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    (tmp_images / "corrupt.jpg").write_bytes(b"not_a_jpeg")

    from src.data.precompute_image_feats import main
    main(tmp_cfg)  # must not raise

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    assert "corrupt" not in feats
    assert len(feats) == 3  # only the 3 valid images


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_no_images_raises(mock_create, tmp_path):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cfg = OmegaConf.create({
        "data": {
            "paths": {
                "images": str(empty_dir),
                "image_feats": str(tmp_path / "out.pt"),
            },
            "image": {"clip_model": "ViT-B-32", "clip_pretrained": "openai"},
        },
        "device": "cpu",
        "batch_size": 64,
    })

    from src.data.precompute_image_feats import main
    with pytest.raises(FileNotFoundError, match="No .jpg"):
        main(cfg)
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_phase2_precompute.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'src.data.precompute_image_feats'`

- [ ] **Step 3: Implement `src/data/precompute_image_feats.py`**

```python
"""Precompute CLIP ViT-B-32 image features and save {stem: Tensor(512)} dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config, resolve_device


def main(cfg: DictConfig) -> None:
    images_dir = Path(cfg.data.paths.images)
    out_path = Path(cfg.data.paths.image_feats)
    batch_size: int = cfg.get("batch_size", 64)
    device = resolve_device(cfg.get("device", "cpu"))

    jpg_paths = sorted(images_dir.glob("*.jpg"))
    if not jpg_paths:
        raise FileNotFoundError(f"No .jpg files found in {images_dir}")
    print(f"Found {len(jpg_paths)} images in {images_dir}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.data.image.clip_model,
        pretrained=cfg.data.image.clip_pretrained,
        device=device,
    )
    model.eval()

    feats_dict: dict[str, torch.Tensor] = {}
    skipped = 0

    for i in tqdm(range(0, len(jpg_paths), batch_size), desc="Encoding"):
        batch_paths = jpg_paths[i : i + batch_size]
        imgs, stems = [], []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                stems.append(p.stem)
            except Exception as exc:
                print(f"Warning: skipping {p.name}: {exc}")
                skipped += 1
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for stem, feat in zip(stems, feats.cpu()):
            feats_dict[stem] = feat

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats_dict, out_path)
    print(f"Saved {len(feats_dict)} features to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP image features")
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    main(cfg)
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
pytest tests/test_phase2_precompute.py -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/precompute_image_feats.py tests/test_phase2_precompute.py
git commit -m "feat(data): add precompute_image_feats script with CLIP ViT-B-32"
```

---

### Task 3: RecipeDataset (TDD)

**Files:**
- Create: `src/data/build_dataset.py`
- Create: `tests/test_phase2_dataset.py`

**Interfaces:**
- Consumes: `kaggle_adapter.load(cfg.data)` → `List[dict]` with keys `id, title, ingredients, instructions, image_path, partition`
- Consumes: `cfg.data.paths.image_feats`, `cfg.data.text.tokenizer`, `cfg.data.text.ingr_max_tokens`, `cfg.data.text.instr_max_tokens`
- Produces:
  - `RecipeDataset(cfg, partition=None)` — `torch.utils.data.Dataset`
  - `RecipeDataset.__getitem__(i) -> dict` — keys: `image_feat, ingr_input_ids, ingr_attention_mask, instr_input_ids, instr_attention_mask, recipe_id, partition`
  - `get_split(cfg, partition) -> RecipeDataset`

- [ ] **Step 1: Write failing tests**

Create `tests/test_phase2_dataset.py`:

```python
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
```

- [ ] **Step 2: Run to verify tests fail**

```bash
pytest tests/test_phase2_dataset.py -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'src.data.build_dataset'`

- [ ] **Step 3: Implement `src/data/build_dataset.py`**

```python
"""RecipeDataset: loads cached image features + tokenizes text on-the-fly."""
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.data import kaggle_adapter


class RecipeDataset(Dataset):
    def __init__(self, cfg: DictConfig, partition: str | None = None) -> None:
        recipes = kaggle_adapter.load(cfg.data)
        if partition is not None:
            recipes = [r for r in recipes if r["partition"] == partition]

        feats_dict: dict[str, torch.Tensor] = torch.load(
            cfg.data.paths.image_feats, weights_only=False
        )

        before = len(recipes)
        recipes = [r for r in recipes if Path(r["image_path"]).stem in feats_dict]
        dropped = before - len(recipes)
        if dropped:
            print(f"Dropped {dropped} recipes missing from image_feats")

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
    return RecipeDataset(cfg, partition=partition)
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
pytest tests/test_phase2_dataset.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
pytest -v
```

Expected: all Phase 0 + Phase 1 + Phase 2 tests PASS (17 + 5 + 8 = 30)

- [ ] **Step 6: Commit**

```bash
git add src/data/build_dataset.py tests/test_phase2_dataset.py
git commit -m "feat(data): add RecipeDataset with cached CLIP feats + on-the-fly tokenization"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|---|---|
| Fix `paths.images` nested dir | Task 1 |
| Add `paths.image_feats` to config | Task 1 |
| `precompute_image_feats.py` — reads `.jpg`, runs CLIP, saves `{stem: Tensor(512)}` | Task 2 |
| Batched with tqdm, configurable batch size | Task 2 |
| Device via `resolve_device` | Task 2 |
| Corrupt image skipped, no images raises FileNotFoundError | Task 2 |
| Creates `data/processed/` if missing | Task 2 |
| `RecipeDataset.__init__` — loads, filters partition, loads feats, inits tokenizer | Task 3 |
| `RecipeDataset.__getitem__` — 7-key dict, correct shapes | Task 3 |
| Recipes missing from feats_dict dropped (not raises) | Task 3 |
| `get_split(cfg, partition)` convenience wrapper | Task 3 |
| Tests: precompute output file, stems, shapes, corrupt skip, no-images error | Task 2 |
| Tests: len, keys, shapes, partition filter, missing feat drop, split counts | Task 3 |

### Placeholder Scan

No TBD/TODO/placeholder patterns found.

### Type Consistency

- `main(cfg: DictConfig)` — used consistently in Task 2 tests and implementation
- `RecipeDataset(cfg, partition=None)` / `get_split(cfg, partition)` — consistent across Task 3 tests and implementation
- `feats_dict: dict[str, Tensor]` — written by Task 2, read by Task 3 via `Path(recipe["image_path"]).stem` — keys match
- `cfg.data` passed to `kaggle_adapter.load()` — matches Phase 1 contract
