# Food Image-to-Recipe Retrieval — Project Overview

## What This Project Does

Given a photo of food, the system finds the most relevant recipes from a database.
This is a **retrieval** problem, not classification — the model doesn't generate recipes,
it searches for the best matching ones using learned embeddings.

Example:
- Input: photo of a pizza
- Output: top-5 most similar recipes ranked by similarity score

---

## The Core Idea: Joint Embedding Space

The key insight: train an image encoder and a recipe text encoder so that
a food photo and its matching recipe land close together in the same vector space.

```
Food Image ──► CLIP ──► projection head ──► 1024-d vector ─┐
                                                             ├─ cosine similarity → rank
Recipe Text ──► DistilBERT ──► fusion head ──► 1024-d vector ─┘
```

At inference: embed query image → cosine search against pre-indexed recipe embeddings → top-K.

---

## Tech Stack & Why Each Tool Was Chosen

### PyTorch
- Framework for all neural network components
- Chosen: industry standard, MPS support for Mac dev, CUDA for Kaggle training

### CLIP ViT-B/32 (via open_clip)
- Encodes food images into 512-d feature vectors
- Chosen: pretrained on 400M image-text pairs — already understands food visually
- Frozen (not fine-tuned): images encoded once and cached → 10x faster training
- Why not fine-tune CLIP: expensive, not needed when projection head can adapt the features
- CLIP text encoder NOT used for recipes: caps at 77 tokens, too short for recipe instructions

### DistilBERT (via HuggingFace transformers)
- Encodes recipe ingredients + instructions into text embeddings
- Chosen: 40% smaller than BERT, 60% faster, retains 97% of BERT's performance
- Handles up to 512 tokens → fits full recipe instructions
- One shared encoder for both ingredient and instruction streams (parameter efficient)

### InfoNCE Loss (NT-Xent / Contrastive Loss)
- Pulls matching image-recipe pairs together, pushes non-matching pairs apart
- Temperature=0.07: sharp similarity distribution, forces the model to be precise
- Symmetric: computed in both directions (image→recipe AND recipe→image)
- Same loss used in CLIP training — proven for cross-modal retrieval

### Fusion Module (3 modes)
- **concat** (baseline): concatenate ingredient + instruction embeddings → linear layer
- **cross-attention**: instructions attend over ingredient tokens (captures "how ingredients interact")
- **ingr_only**: ablation to measure how much instructions add vs ingredients alone

### OmegaConf + YAML configs
- All hyperparameters in `configs/` — no hardcoded values in source code
- CLI overrides: `train.lr=5e-5` on the command line, no file editing needed
- Enables reproducible experiments (different configs = different runs)

### faiss-cpu (for future scaling)
- Fast approximate nearest neighbor search for large recipe databases
- CPU version because faiss-gpu unavailable on Mac (Apple Silicon)

### TensorBoard
- Logs training loss + validation metrics per epoch
- View with: `tensorboard --logdir runs/`

---

## Data Pipeline

```
Kaggle CSV (13k recipes + image names)
         │
         ▼
kaggle_adapter.py
  - filters recipes with no image
  - parses ingredient lists
  - assigns 80/10/10 train/val/test split
  - samples n_recipes (default 2000 for smoke, 10k+ for real run)
         │
         ▼
precompute_image_feats.py
  - runs CLIP ViT-B/32 on all 13,582 images
  - saves {image_stem: Tensor(512)} to data/processed/image_feats.pt
  - runs ONCE, reused across all training runs
         │
         ▼
RecipeDataset (build_dataset.py)
  - loads cached CLIP features from .pt file
  - tokenizes ingredients + instructions on-the-fly with DistilBERT tokenizer
  - returns 7-key dict per sample: image_feat, ingr_input_ids, ingr_attention_mask,
    instr_input_ids, instr_attention_mask, recipe_id, partition
```

---

## Model Architecture

```
                    ┌─────────────────────────────────────┐
                    │         JointEmbeddingModel          │
                    │                                      │
  image_feat(512) ──► ImageEncoder                        │
                    │   Linear(512→1024) + LayerNorm       │
                    │   GELU                               │
                    │   Linear(1024→1024) + LayerNorm      │
                    │   L2-normalize              image_emb(1024)
                    │                                      │
  ingr_tokens ─────► TextEncoder (DistilBERT)             │
  instr_tokens ────►   masked mean pool                   │
                    │   separate linear heads (768→1024)  │
                    │         │                            │
                    │         ▼                            │
                    │   FusionModule (concat mode)         │
                    │   [ingr_emb; instr_emb] → Linear    │
                    │   L2-normalize             recipe_emb(1024)
                    └─────────────────────────────────────┘
```

---

## Training Loop

1. Load batch: (image_feat, ingr_tokens, instr_tokens)
2. Forward pass → (image_emb, recipe_emb) both shape (B, 1024), L2-normalized
3. Compute InfoNCE loss: B×B similarity matrix, diagonal = positives
4. Backward pass with gradient accumulation
5. Validate every epoch: compute medR + R@1/5/10 on val set
6. Save checkpoint if val medR improves (lower = better)
7. Early stop if no improvement for 5 epochs

---

## Evaluation Metrics

Both directions measured at both 1k and 10k recipe subsets:

- **medR (Median Rank)**: median position of the correct recipe in ranked results
  - Perfect = 1.0, Random on 1k = 500 — lower is better
- **R@1**: % of queries where correct recipe is #1 result
- **R@5**: % of queries where correct recipe is in top-5
- **R@10**: % of queries where correct recipe is in top-10

Why both directions:
- `im2recipe`: query=image, search=recipes
- `recipe2im`: query=recipe, search=images
A good model should work both ways.

---

## File Map

```
configs/
  baseline.yaml       ← main config (fusion=concat, no attention)
  fusion.yaml         ← config with cross-attention fusion
  data.yaml           ← dataset paths, split fractions, tokenizer settings

src/
  data/
    kaggle_adapter.py       ← load + parse CSV, assign splits
    precompute_image_feats.py ← CLIP encode all images → .pt cache
    build_dataset.py        ← RecipeDataset + get_split()
  models/
    image_encoder.py        ← MLP projection: 512 → 1024, L2-norm
    text_encoder.py         ← DistilBERT, masked mean pool, ingr+instr heads
    fusion.py               ← concat | cross-attention | ingr_only
    joint_embedding.py      ← top-level model combining all above
  losses/
    infonce.py              ← symmetric NT-Xent, temperature=0.07
  eval/
    metrics.py              ← medR + R@k, both directions
    evaluate.py             ← eval a checkpoint on test/val split
    demo.py                 ← query top-K recipes from a single image
  training/
    train.py                ← full train loop: AMP, grad accum, early stop, TensorBoard
  utils/
    config.py               ← OmegaConf loader + resolve_device()
    seed.py                 ← set_seed() for reproducibility

data/
  raw/                      ← CSV + images (gitignored)
  processed/image_feats.pt  ← cached CLIP features (gitignored)

runs/
  baseline_concat/
    best.pt                 ← saved checkpoint (best val medR)
    events.out.*            ← TensorBoard logs

tests/                      ← 58 tests covering all components
```

---

## Key Design Decisions

### Why cache CLIP features instead of running CLIP each batch?
CLIP ViT-B/32 is expensive to run. With 13k images and 30 epochs, running CLIP
per-batch would repeat the same computation 30× per image. Cache once → 10x faster.

### Why DistilBERT for text, not CLIP's text encoder?
CLIP text encoder caps at 77 tokens — recipes average 200+ tokens for instructions.
DistilBERT handles 512 tokens and is trained on general text.

### Why separate ingredient + instruction streams?
Ingredients and instructions are structurally different:
- Ingredients: unordered list (order doesn't matter → mean pooling)
- Instructions: sequential steps (order matters → full sequence encoding)
Separating them allows the fusion module to weight them differently.

### Why InfoNCE loss?
It's what CLIP itself uses. For cross-modal retrieval, it's proven to work well.
The temperature parameter (0.07) controls how "sharp" the similarity scores are —
lower temperature = model must be more certain about the correct match.

### Why 1024-d embedding space?
Large enough to capture visual + textual semantics, small enough for fast cosine search.

---

## How to Run

```bash
# 1. Precompute CLIP features (once)
python -m src.data.precompute_image_feats --config baseline.yaml

# 2. Train
python -m src.training.train --config baseline.yaml train.num_workers=0 train.batch_size=32

# 3. Evaluate on test set
python -m src.eval.evaluate --config baseline.yaml --checkpoint runs/baseline_concat/best.pt

# 4. Demo: query with a single image
python -m src.eval.demo \
  --config baseline.yaml \
  --checkpoint runs/baseline_concat/best.pt \
  --image "path/to/food.jpg" \
  --topk 5 \
  --split train

# 5. View training curves
tensorboard --logdir runs/
```

---

## Current Results (Local Mac, 2000 recipes, 6 epochs)

- val im2recipe medR = 10.0 (out of 200 val recipes)
- All top-5 predictions for pizza image = pizza variants (semantically correct)
- Ground truth found at rank #5 for pizza test

Expected on Kaggle (full dataset, more epochs): medR ~5-10 on 1k subset, R@10 > 50%.

---

## What Makes This Hard

1. **Cross-modal gap**: images and text live in completely different feature spaces
2. **Recipe ambiguity**: many dishes look similar (pasta dishes, salads, soups)
3. **Ingredient order**: ingredient lists have no meaningful order — must use order-invariant pooling
4. **Scale**: evaluating retrieval on 10k recipes requires efficient similarity search
