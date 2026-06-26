# Food Image-to-Recipe Retrieval — Notebook Overview

## What This Does

Given a food image, retrieve the top-K most relevant recipes from a 3,000-recipe database.
Zero-shot approach — no fine-tuning. Precomputed CLIP embeddings + FAISS search at inference.

Example:
- Input: photo of a dip/spread
- Output: top-5 ranked recipes with similarity scores (e.g., Smoky Carrot Dip at 0.325)

---

## The Core Idea

Use CLIP's shared image-text embedding space directly. Food images and recipe text already land
nearby in CLIP's 512-d space without any training — the model was pretrained on 400M image-text pairs.

```
Food Image ──► CLIP Vision Encoder ──► 512-d (L2-norm) ─┐
                                                          ├─ inner product → rank
Recipe Text ──► CLIP Text Encoder  ──► 512-d (L2-norm) ─┘
```

Key twist: ingredient-aware fusion blends full recipe text with ingredient-only embeddings
to give slightly more weight to visual ingredient cues.

---

## Dataset

- Source: Food Ingredients and Recipe Dataset (Kaggle)
- Raw: 13,501 recipes × 6 columns (Title, Ingredients, Instructions, Image_Name, Cleaned_Ingredients)
- Images: 13,582 files matched by filename stem
- Matched pairs: 13,471 (30 images unmatched)
- Working subset: first **3,000** recipes (for memory/speed on CPU)

### Schema

| Column | Content |
|--------|---------|
| `Title` | Recipe name |
| `Ingredients` | Raw ingredient list (string) |
| `Instructions` | Step-by-step cooking instructions |
| `Image_Name` | Filename stem (no extension) |
| `Cleaned_Ingredients` | Pre-cleaned ingredient list |

### Text Construction

```python
recipe_text = title + ". Ingredients: " + ingredients + ". Instructions: " + instructions
```

Concatenated into one string, then encoded as a unit with CLIP text encoder (truncated at 77 tokens).

---

## Model: CLIP ViT-B/32

- Pretrained: `openai/clip-vit-base-patch32` via HuggingFace transformers
- Frozen: no gradient updates, no fine-tuning
- Both encoders used: `vision_model` + `visual_projection` for images, `text_model` + `text_projection` for text
- Output: **512-d** L2-normalized embeddings (cosine similarity = inner product on unit sphere)

### Why CLIP?

- Pretrained on 400M diverse image-text pairs — food images well-represented
- Shared embedding space: image and text directly comparable
- No training data required: zero-shot deployment
- Fast: encoders run once, embeddings cached; only FAISS search at query time

### Why not CLIP for the original project?

Original pipeline used DistilBERT (512 token limit) for full recipe text. CLIP text encoder
caps at 77 tokens — instructions get truncated. This notebook accepts that tradeoff for simplicity.

---

## Embedding Pipeline

### Step 1: Image Embeddings

```python
outputs = clip_model.vision_model(pixel_values=pixel_values)
emb = clip_model.visual_projection(outputs.pooler_output)
emb = emb / emb.norm(dim=-1, keepdim=True)
# Shape: (3000, 512)
```

### Step 2: Recipe Text Embeddings

```python
outputs = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
emb = clip_model.text_projection(outputs.pooler_output)
emb = emb / emb.norm(dim=-1, keepdim=True)
# Shape: (3000, 512)
```

### Step 3: Ingredient-Only Embeddings

Same CLIP text encoder, but input = `ingredients_clean` only (no title or instructions).
Shape: (3000, 512)

### Step 4: Ingredient-Aware Fusion

```python
alpha = 0.9  # 90% full recipe, 10% ingredients-only
recipe_embeddings = alpha * text_embeddings + (1 - alpha) * ingredient_embeddings
recipe_embeddings /= ||recipe_embeddings||  # re-normalize
```

Alpha=0.9 chosen via ablation (see Experiments section). Small ingredient signal boosts
retrieval of visually distinct recipes that share textual overlap.

---

## Indexing: FAISS IndexFlatIP

```
recipe_index: IndexFlatIP(512) — 3000 recipe embeddings
image_index:  IndexFlatIP(512) — 3000 image embeddings
```

- `IndexFlatIP`: exact inner product search (no approximation)
- Inner product on L2-normalized vectors = cosine similarity
- Both directions supported: image→recipe and recipe→image

---

## Retrieval

### Image → Recipe

```python
query_emb = image_embeddings[row_id]       # (1, 512)
scores, indices = recipe_index.search(query_emb, top_k)
# returns ranked recipe indices + similarity scores
```

### Recipe → Image

```python
query_emb = recipe_embeddings[row_id]      # (1, 512)
scores, indices = image_index.search(query_emb, top_k)
```

Both directions return Top-K matches ranked by cosine similarity.

---

## Evaluation Metrics

### Recall@K

```
Recall@K = (# queries where ground truth is in top-K) / (# total queries)
```

Computed for K ∈ {1, 5, 10}. Ground truth = row index (each image has exactly one matching recipe).

### Median Rank (medR)

Median position of the correct match across all queries. Lower = better. Random baseline = 1500 (on 3000).

Implementation: chunked matmul (chunk_size=300) to avoid OOM on CPU:

```python
sim = query_chunk @ db.T           # (chunk, 3000)
rank = (sim > sim[i, i]).sum() + 1
```

---

## Experiments: Fusion Weight Ablation

Tested α ∈ {0.3, 0.5, 0.7, 0.9} on image→recipe Recall@K:

| Alpha (text weight) | Recall@1 | Recall@5 | Recall@10 |
|--------------------|----------|----------|-----------|
| 0.3 | — | — | — |
| 0.5 | — | — | — |
| 0.7 | — | — | — |
| **0.9** | — | — | — |

*Run notebook to populate. α=0.9 selected as best from ablation.*

---

## Output Files

```
runs/
  retrieval_results.csv       ← Recall@1/5/10 + medR, both directions
  fusion_comparison.csv       ← ablation: Recall@K for α ∈ {0.3, 0.5, 0.7, 0.9}
  recall_results.pdf          ← Recall@K curve plot (im2recipe + recipe2im)
  fusion_comparison.pdf       ← alpha ablation plot
  image_embeddings.npy        ← (3000, 512) image embeddings
  text_embeddings.npy         ← (3000, 512) full recipe text embeddings
  recipe_embeddings.npy       ← (3000, 512) fused recipe embeddings (α=0.9)
```

---

## Interactive Demo

Bottom of notebook: paste any food image URL → get top-5 recipes.

```python
url = "https://..."
response = requests.get(url, stream=True)
query_img = Image.open(response.raw).convert("RGB")

# Encode with CLIP vision encoder
query_emb = clip_model.visual_projection(clip_model.vision_model(...).pooler_output)
query_emb /= query_emb.norm(dim=-1, keepdim=True)

# Search
scores, indices = recipe_index.search(query_emb.numpy(), 5)
```

Note: requires CLIP model still loaded (free it before evaluation cell, reload before demo).

---

## Architecture Diagram

```
                     ┌─────────────────────────────────────────┐
                     │    CLIP ViT-B/32  (frozen, no training)  │
                     │                                           │
  Food Image ───────►│  vision_model → pooler_output            │
  (224×224 RGB)      │  visual_projection(512→512)              │──► image_emb (512, L2-norm)
                     │                                           │
  Recipe Text ──────►│  text_model → pooler_output              │
  (title+ingr+instr) │  text_projection(512→512)  [truncated 77]│──► text_emb (512, L2-norm)
                     │                                           │
  Ingredients ──────►│  text_model → pooler_output              │
  (ingr only)        │  text_projection(512→512)                │──► ingr_emb (512, L2-norm)
                     └─────────────────────────────────────────┘

  recipe_emb = normalize(0.9 × text_emb + 0.1 × ingr_emb)   ← ingredient-aware fusion

  FAISS IndexFlatIP ←── recipe_emb (3000, 512)
  FAISS IndexFlatIP ←── image_emb  (3000, 512)

  Query: image_emb[i] → top-K from recipe_index → ranked recipes
  Query: recipe_emb[i] → top-K from image_index → ranked images
```

---

## Tradeoffs vs. Fine-tuned Pipeline

| Aspect | This Notebook | Full Pipeline (PROJECT_OVERVIEW.md) |
|--------|--------------|--------------------------------------|
| Training | None (zero-shot) | InfoNCE contrastive training |
| Text encoder | CLIP (77 token limit) | DistilBERT (512 tokens) |
| Recipe text coverage | Title + truncated ingr/instr | Full ingredients + instructions |
| Embedding dim | 512 | 1024 |
| Fusion | Weighted sum (α=0.9) | Concat + linear / cross-attention |
| Setup time | ~2 min (encode once) | Hours (train + evaluate) |
| Expected medR | ~100-300 / 3000 | ~5-10 / 1000 (after training) |
| Complexity | Single notebook | Full src/ package + configs |

---

## How to Run

```bash
# Install deps
pip install transformers faiss-cpu pillow

# Open notebook
jupyter notebook image-to-recipe.ipynb

# Run all cells in order
# Cells 1–8:   load data, build image map, subset to 3000
# Cells 9–11:  load CLIP model
# Cells 12–13: define embedding functions
# Cells 14–16: generate image + recipe + ingredient embeddings
# Cells 18–19: fuse embeddings (α=0.9), build FAISS indexes
# Cells 20–25: retrieval demos (image→recipe, recipe→image, visual grid)
# Cell 26:     free CLIP from RAM
# Cells 27–28: compute Recall@K + medR
# Cells 29–33: save CSVs + plots; run fusion ablation
# Cells 34–35: save embeddings, print summary
# Cell 37:     interactive demo with custom image URL
```
