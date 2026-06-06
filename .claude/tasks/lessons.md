# lessons.md — Food Image-to-Recipe Retrieval

Permanent rules learned during this project. Review at session start.
Each entry: what happened → rule to prevent repeat.

---

## Environment / compute

- **M5 MacBook Air, 24GB unified RAM = DEV box, not training box.**
  Apple Silicon → no CUDA → PyTorch uses MPS (works, slower/flakier than NVIDIA).
  `faiss-gpu` unavailable on Mac; use `faiss-cpu`.
  Rule: write + debug on Mac (tiny fixtures, MPS sanity), train + eval on Kaggle GPU.

- **Kaggle = training horse:** free T4×2 (16GB ea) or P100, ~30 GPU-hr/week, internet on.
  Rule: respect weekly GPU budget; cache expensive feats once, never recompute per run.

## Data

- **Full Recipe1M images ≈ 330GB.** Won't fit laptop or "a project file."
  Rule: only need `layer1.json` (recipes), `layer2.json` (id→images), `det_ingrs.json`,
  plus a SUBSET of images for the sampled recipes. Never download the full image tarball.
- **NEVER commit `data/` or `runs/`.** Gitignore them. Data reproducible from `data/raw/`.
- **Recipe1M `layer1.json` entries carry a `partition` field** (train/val/test) — use it as the
  canonical split. Never let a recipe id leak across splits.

## Modeling decisions (the parts generic agents get wrong)

- **CLIP ViT-B/32 text encoder caps at 77 tokens** → too short for recipe instructions.
  Rule: CLIP for IMAGES only (512-d). Recipe text → DistilBERT (512 tokens, 768-d).
- **Image tower frozen + cached** (the expensive part). Caching CLIP image feats once → ~10x faster training.
- **Ingredient list order is NOT semantically meaningful.**
  Rule: permutation-robust pooling (attention or mean), never position-dependent.
- **Report BOTH directions (im2recipe + recipe2im) at BOTH subsets (1k + 10k).** A single direction is incomplete.
- **Single seed is not a result.** Headline numbers = mean ± std over ≥3 seeds.
- **Eval must be deterministic** given (ckpt, subset, seed). medR moving without a code change = bug.

## Process

- (seed) Plan before build; verify before done. Add lessons here after every correction.
