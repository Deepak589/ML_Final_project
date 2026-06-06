# AGENTS.md — Food Image-to-Recipe Retrieval with Ingredient-Aware Fusion

> Operating manual for AI coding agents on this repo. Read fully before editing.
> Optimized for: correctness, reproducibility, fast iteration on a Hard ML project.

## 1. Project summary

Cross-modal retrieval between **food images** and **recipes** (title + ingredients + instructions).
Train a shared embedding space so an image embedding is near its matching recipe embedding.
Evaluate **both directions**: image→recipe (im2recipe) and recipe→image (recipe2im).
The "ingredient-aware fusion" twist: ingredients are encoded as a separate stream and fused
with instruction/title text via attention before joining the image stream.

- **Dataset:** Recipe1M / Recipe1M+ — http://pic2recipe.csail.mit.edu/
- **Baseline to beat:** im2recipe joint embedding (Salvador et al., CVPR 2017).
- **Primary metrics:** median rank (medR, lower=better) and Recall@{1,5,10} (R@K, higher=better),
  reported on the standard **1k** and **10k** random test subsets, both retrieval directions.

## 2. Commands (use these exact ones)

```bash
# env (Python 3.10+, CUDA 12.x). Pin everything in requirements.txt / environment.yml
pip install -r requirements.txt

# data prep (run once; writes LMDB/HDF5 shards + vocab to data/processed/)
python -m src.data.build_dataset --config configs/data.yaml

# precompute image features (cache ResNet/CLIP features; ~10x faster training)
python -m src.data.precompute_image_feats --split train --config configs/data.yaml

# train (mixed precision + checkpointing; resumes from last ckpt automatically)
python -m src.train --config configs/baseline.yaml

# evaluate retrieval (FAISS-backed; fixed seed -> comparable medR across runs)
python -m src.eval --ckpt runs/<exp>/best.pt --subset 1k   --direction both
python -m src.eval --ckpt runs/<exp>/best.pt --subset 10k  --direction both

# tests + lint + format (must pass before any commit)
pytest -q
ruff check . && ruff format --check .
```

Always pass `--config`. Never hardcode paths, hyperparams, or seeds in source — they live in `configs/*.yaml`.

## 3. Tech stack (specific versions, no "just use latest")

- **PyTorch 2.x** + torchvision, CUDA 12.x, AMP (`torch.cuda.amp`) for mixed precision.
- **Image encoder:** ResNet-50 (im2recipe parity) OR CLIP Vi-B/32 image tower (stronger; A/B both).
- **Text encoders:** HuggingFace `transformers` (DistilBERT/BERT) for ingredients + instructions,
  OR LSTM for strict im2recipe reproduction. Keep encoders swappable behind a common interface.
- **Retrieval/eval:** `faiss-cpu` (or `faiss-gpu`) for ANN search at 10k+ scale. Do NOT do O(N²) numpy.
- **Data IO:** LMDB or HDF5 shards. `webdataset` acceptable for Recipe1M+ scale.
- **Tracking:** Weights & Biases or TensorBoard (configurable). Log loss, medR, R@K per eval.
- **Config:** Hydra or OmegaConf YAML. CLI flags override config keys only.

## 4. Project structure

```
configs/        # YAML: data.yaml, baseline.yaml, fusion.yaml — single source of truth for params
data/
  raw/          # original Recipe1M(+) downloads — NEVER edit, NEVER commit
  processed/    # generated LMDB/HDF5 + vocab.pkl — gitignored, reproducible from raw
src/
  data/         # build_dataset, dataset classes, tokenizers, precompute_image_feats
  models/       # image_encoder.py, ingredient_encoder.py, instruction_encoder.py,
                # fusion.py (attention fusion), joint_model.py
  losses/       # triplet.py, infonce.py, semantic_reg.py
  train.py      # training loop (AMP, grad-accum, ckpt, early stop on val medR)
  eval.py       # FAISS retrieval eval, 1k/10k, both directions
  utils/        # seed.py, metrics.py (medR, R@K), io.py
tests/          # unit tests; fast, no network, tiny fixtures
runs/           # experiment outputs/checkpoints — gitignored
notebooks/      # exploration only; never import notebooks from src/
```

## 5. ML methodology rules (the part generic agents get wrong)

**Embedding & loss**
- Project image and recipe to the **same dim** (e.g. 1024), L2-normalize, compare with cosine.
- Default loss: **InfoNCE / contrastive** with in-batch negatives (large batch helps). Triplet loss
  with hard-negative mining is the im2recipe baseline — implement both, make it a config switch.
- Keep the **semantic regularization** head (food-category classifier aux loss) from the baseline;
  it measurably improves medR. Weight it via config (`lambda_sem`).

**Ingredient-aware fusion (the project's core contribution)**
- Encode ingredients and instructions as separate streams; fuse with cross-/self-attention in
  `fusion.py` before the final recipe projection. Ablate: (a) concat, (b) attention fusion,
  (c) ingredients-only — report all three.
- Ingredient list order is not semantically meaningful — use permutation-robust pooling
  (attention or mean), not position-dependent assumptions.

**Data discipline**
- Iterate on the **Recipe1M (1M)** split first; only run full **Recipe1M+ (~13M images)** for final numbers.
- Use the **canonical train/val/test split**. Never let a recipe id leak across splits.
- Fix the 1k/10k test sampling with a **fixed seed** so medR is comparable run-to-run and to papers.

## 6. Reproducibility (non-negotiable)

- Set and log seeds everywhere (`src/utils/seed.py`): python, numpy, torch, cudnn deterministic for eval.
- Every run writes its resolved config + git SHA into `runs/<exp>/config_used.yaml`.
- Report mean ± std over ≥3 seeds for any headline result. A single run is not a result.
- Eval must be deterministic given (ckpt, subset, seed). If medR moves without a code change, that's a bug.

## 7. Code style

- Type hints on all public functions. Docstrings state tensor **shapes + dtypes**, e.g. `(B, D) float32`.
- No magic numbers in `src/` — pull from config. Functions small and pure where possible.
- Log shapes at module boundaries during dev; assert shapes in `forward()`.

```python
# ✅ Good — shapes documented, config-driven, normalized
def project(self, x: Tensor) -> Tensor:  # x: (B, in_dim) -> (B, D) L2-normalized
    return F.normalize(self.mlp(x), dim=-1)

# ❌ Bad — magic dims, no norm, silent shape assumptions
def f(self, x):
    return self.mlp(x)  # 1024? normalized? who knows
```

## 8. Git workflow

- Branch per experiment: `exp/fusion-attention`, `fix/eval-seed`. Small, reviewable commits.
- Commit messages state **what changed and the metric impact** when relevant
  (e.g. `feat(fusion): attention fusion, 10k im2recipe medR 5.0 -> 4.2`).
- Never commit to `main` directly. Run `pytest -q` and lint before every commit.

## 9. Boundaries

- ✅ **Always:** write to `src/`, `tests/`, `configs/`; keep `data/processed` reproducible from `data/raw`;
  fix + log seeds; report both retrieval directions at 1k and 10k.
- ⚠️ **Ask first:** changing the canonical data split, swapping the loss/encoder default,
  adding a heavy dependency, or anything that touches `configs/baseline.yaml` defaults.
- 🚫 **Never:** commit `data/`, `runs/`, checkpoints, or large binaries (use `.gitignore`);
  edit files in `data/raw/`; tune on or peek at the test set; report a number from a single seed
  or an undocumented eval setup; fabricate metrics — if a run didn't finish, say so.

## 10. Definition of done (per task)

A change is done only when: tests pass, lint clean, config (not code) holds new params,
the affected metric is re-evaluated with the fixed-seed protocol, and the result (with seed
count and subset) is recorded in the experiment log / commit message.
