# image-to-recipe.ipynb Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch `image-to-recipe.ipynb` to fix the MiniLM zero-padding hack, enable CUDA on Kaggle T4, set α=0.9, add medR metric, and add an interactive URL demo cell.

**Architecture:** Zero-shot retrieval using CLIP ViT-B/32 for both image and text (ingredients + recipe). All embeddings live in the same 512-dim CLIP space — no cross-model dimension mismatch. FAISS IndexFlatIP for retrieval. No training loop.

**Tech Stack:** `transformers`, `faiss-cpu`, `torch`, `PIL`, `requests`, `numpy`, `pandas`, `matplotlib`

## Global Constraints

- File to edit: `image-to-recipe.ipynb` (project root)
- Dataset cap: 3000 rows (unchanged — `df.head(3000)`)
- Model: `openai/clip-vit-base-patch32` (unchanged)
- No new pip packages — remove `sentence-transformers` from install cell
- FAISS index type: `IndexFlatIP` (unchanged)
- All cell edits use `NotebookEdit` tool with exact `cell_id` and `new_source`

---

### Task 1: Fix install cell + imports + device

**Files:**
- Modify: `image-to-recipe.ipynb` cells: `cell-0`, `cell-1`, `cell-2`

**Interfaces:**
- Produces: `device` variable (`"cuda"` on T4, `"cpu"` locally) used by all subsequent cells

- [ ] **Step 1: Remove `sentence-transformers` from install cell (`cell-0`)**

Edit `cell-0` new source:
```python
!pip install -q transformers faiss-cpu
```

- [ ] **Step 2: Clean up imports in `cell-1`**

Edit `cell-1` new source:
```python
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import faiss

from transformers import CLIPProcessor, CLIPModel
```

(Removed: `SentenceTransformer`, `PCA` — both unused after this patch.)

- [ ] **Step 3: Fix device detection in `cell-2`**

Edit `cell-2` new source:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)
```

- [ ] **Step 4: Verify by running cells 0-2 on Kaggle**

Expected output from cell-2:
```
Torch version: 2.x.x+cu...
CUDA available: True
Using device: cuda
```

- [ ] **Step 5: Commit**

```bash
git add image-to-recipe.ipynb
git commit -m "fix: use CUDA when available, drop sentence-transformers install and imports"
```

---

### Task 2: Replace MiniLM ingredient embedding with CLIP text encoder

**Files:**
- Modify: `image-to-recipe.ipynb` cells: `cell-11`, `cell-16`, `cell-17`

**Interfaces:**
- Consumes: `get_clip_text_embeddings()` defined in `cell-13`, `df["ingredients_clean"]` from `cell-6`
- Produces: `ingredient_embeddings` — shape `(3000, 512)`, L2-normalized, in CLIP text embedding space (replaces the zero-padded 384→512 hack)

- [ ] **Step 1: Delete the MiniLM model load cell (`cell-11`)**

Edit `cell-11` new source (replace with a comment marker so cell still exists but does nothing):
```python
# Ingredient encoder: using CLIP text encoder (cell-13) — no separate model needed
```

- [ ] **Step 2: Replace ingredient embedding cell (`cell-16`) to use CLIP**

Edit `cell-16` new source:
```python
# ============================================================
# 16. Generate Ingredient Embeddings via CLIP Text Encoder
# ============================================================

ingredient_embeddings = get_clip_text_embeddings(
    df["ingredients_clean"].tolist(),
    batch_size=8
)

print("Ingredient embeddings:", ingredient_embeddings.shape)
# Expected: (3000, 512) — same space as text_embeddings, no padding needed
```

- [ ] **Step 3: Delete the zero-padding cell (`cell-17`)**

Edit `cell-17` new source:
```python
# Zero-padding removed: ingredient_embeddings already 512-dim from CLIP text encoder
```

- [ ] **Step 4: Verify shapes in Kaggle output**

After running cells 14-17, expected output:
```
Image embeddings: (3000, 512)
Text embeddings: (3000, 512)
Ingredient embeddings: (3000, 512)
```
All three are 512-dim. No padding cell needed.

- [ ] **Step 5: Commit**

```bash
git add image-to-recipe.ipynb
git commit -m "fix: replace MiniLM+zero-padding with CLIP text encoder for ingredient embeddings"
```

---

### Task 3: Update fusion weight to α=0.9

**Files:**
- Modify: `image-to-recipe.ipynb` cell: `cell-18`

**Interfaces:**
- Consumes: `text_embeddings (3000, 512)`, `ingredient_embeddings (3000, 512)` from Tasks 1-2
- Produces: `recipe_embeddings (3000, 512)` — fused, L2-normalized

- [ ] **Step 1: Update fusion alpha in `cell-18`**

Edit `cell-18` new source:
```python
# ============================================================
# 18. Ingredient-Aware Fusion  (α=0.9 — best from ablation)
# ============================================================

alpha = 0.9  # 90% full recipe text, 10% ingredient-only signal

recipe_embeddings = (
    alpha * text_embeddings +
    (1 - alpha) * ingredient_embeddings
)

recipe_embeddings = recipe_embeddings / np.linalg.norm(
    recipe_embeddings,
    axis=1,
    keepdims=True
)

print("Final recipe embeddings:", recipe_embeddings.shape)
```

- [ ] **Step 2: Commit**

```bash
git add image-to-recipe.ipynb
git commit -m "fix: set fusion alpha=0.9 (best from ablation: R@1=21.6%, R@10=59.2%)"
```

---

### Task 4: Add medR metric to evaluation

**Files:**
- Modify: `image-to-recipe.ipynb` cells: `cell-26`, `cell-27`

**Interfaces:**
- Consumes: `image_embeddings (3000, 512)`, `recipe_embeddings (3000, 512)`, `recipe_index`, `image_index`
- Produces: results table with columns `K`, `Image_to_Recipe_Recall`, `Recipe_to_Image_Recall`, `Image_to_Recipe_medR`, `Recipe_to_Image_medR`

- [ ] **Step 1: Add `median_rank` function alongside `recall_at_k` in `cell-26`**

Edit `cell-26` new source:
```python
# ============================================================
# 23. Recall@K and Median Rank Evaluation
# ============================================================

def recall_at_k(query_embeddings, index, k):
    scores, indices = index.search(query_embeddings.astype("float32"), k)
    correct = 0
    for i in range(len(query_embeddings)):
        if i in indices[i]:
            correct += 1
    return correct / len(query_embeddings)


def median_rank(query_embeddings, index):
    n = len(query_embeddings)
    scores, indices = index.search(query_embeddings.astype("float32"), n)
    ranks = []
    for i in range(n):
        rank_positions = np.where(indices[i] == i)[0]
        rank = int(rank_positions[0]) + 1 if len(rank_positions) > 0 else n + 1
        ranks.append(rank)
    return float(np.median(ranks))
```

- [ ] **Step 2: Update results cell (`cell-27`) to include medR**

Edit `cell-27` new source:
```python
results = []

im2r_medR = median_rank(image_embeddings, recipe_index)
r2im_medR = median_rank(recipe_embeddings, image_index)

for k in [1, 5, 10]:
    image_to_recipe = recall_at_k(image_embeddings, recipe_index, k)
    recipe_to_image = recall_at_k(recipe_embeddings, image_index, k)
    results.append({
        "K": k,
        "Image_to_Recipe_Recall": round(image_to_recipe, 4),
        "Recipe_to_Image_Recall": round(recipe_to_image, 4),
    })

results_df = pd.DataFrame(results)
results_df["Image_to_Recipe_medR"] = im2r_medR
results_df["Recipe_to_Image_medR"] = r2im_medR

print(f"Median Rank  Im→Recipe: {im2r_medR:.1f}   Recipe→Im: {r2im_medR:.1f}")
results_df
```

- [ ] **Step 3: Verify output on Kaggle**

Expected output (approximate, α=0.9):
```
Median Rank  Im→Recipe: ~X.X   Recipe→Im: ~X.X

   K  Image_to_Recipe_Recall  Recipe_to_Image_Recall  Image_to_Recipe_medR  Recipe_to_Image_medR
0  1                  0.2163                  0.xxxx                  XX.X                  XX.X
1  5                  0.4757                  0.xxxx                  XX.X                  XX.X
2 10                  0.5920                  0.xxxx                  XX.X                  XX.X
```

- [ ] **Step 4: Commit**

```bash
git add image-to-recipe.ipynb
git commit -m "feat: add median rank metric to retrieval evaluation"
```

---

### Task 5: Add interactive URL demo cell

**Files:**
- Modify: `image-to-recipe.ipynb` — append new cell after `cell-35` (final cell)

**Interfaces:**
- Consumes: `image_embeddings`, `recipe_index`, `clip_model`, `clip_processor`, `device`, `df`, `title_col`, `ingredients_col` — all defined in prior cells
- Produces: printed recipe results + displayed query image in Kaggle output

- [ ] **Step 1: Add interactive demo cell at end of notebook**

Add new cell with source:
```python
# ============================================================
# INTERACTIVE DEMO — paste any food image URL below
# ============================================================
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Spaghetti_bolognese_%28hozinja%29.jpg/800px-Spaghetti_bolognese_%28hozinja%29.jpg"

# Fetch and display query image
response = requests.get(url, stream=True, timeout=10)
query_img = Image.open(response.raw).convert("RGB")

plt.figure(figsize=(4, 4))
plt.imshow(query_img)
plt.axis("off")
plt.title("Query Image")
plt.show()

# Encode query image with CLIP
inputs = clip_processor(images=query_img, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

with torch.no_grad():
    outputs = clip_model.vision_model(pixel_values=pixel_values)
    pooled = outputs.pooler_output
    query_emb = clip_model.visual_projection(pooled)
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

query_np = query_emb.cpu().numpy().astype("float32")

# Search top-5 recipes
scores, indices = recipe_index.search(query_np, 5)

print("\nTop-5 Retrieved Recipes:\n" + "=" * 50)
for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
    print(f"\n#{rank}  {df.loc[idx, title_col]}  (score: {score:.3f})")
    print(f"    {str(df.loc[idx, ingredients_col])[:200]}...")
```

- [ ] **Step 2: Test on Kaggle with default spaghetti URL**

Expected: Query image displays, 5 pasta/Italian recipes printed with similarity scores.

- [ ] **Step 3: Verify user can swap URL**

Change `url` to a pizza image URL and re-run — should return pizza/Italian recipes.

- [ ] **Step 4: Commit**

```bash
git add image-to-recipe.ipynb
git commit -m "feat: add interactive food image URL demo cell"
```

---

## Self-Review

**Spec coverage:**
- ✅ Fix device → Task 1
- ✅ Remove MiniLM + zero-padding → Task 2
- ✅ Use CLIP text for ingredients → Task 2
- ✅ α=0.9 → Task 3
- ✅ medR metric → Task 4
- ✅ Interactive URL demo cell → Task 5
- ✅ No new pip installs (sentence-transformers removed) → Task 1

**Placeholder scan:** No TBDs, all code blocks complete.

**Type consistency:** `ingredient_embeddings` is `(3000, 512) np.ndarray` throughout Tasks 2→3. `recipe_index` is `faiss.IndexFlatIP` consumed in Tasks 4→5. All consistent.
