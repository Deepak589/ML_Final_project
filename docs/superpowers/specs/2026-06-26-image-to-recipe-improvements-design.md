# image-to-recipe.ipynb Improvements Design

**Date:** 2026-06-26  
**Scope:** Zero-shot retrieval notebook improvements for Kaggle T4 GPU presentation  
**Cap:** 3000 images (unchanged)  
**Approach:** Minimal patch (Option A)

---

## Goal

Fix broken/suboptimal parts of `image-to-recipe.ipynb` without changing the zero-shot retrieval architecture. Result must run cleanly on Kaggle T4 and be presentation-ready.

---

## Changes to Existing Cells

### cell-2 — Device detection
- **Before:** `device = "cpu"` (hardcoded, wastes T4 GPU)
- **After:** `device = "cuda" if torch.cuda.is_available() else "cpu"`

### cell-1 — Imports
- Remove `from sentence_transformers import SentenceTransformer` (deleted model)
- Remove `from sklearn.decomposition import PCA` (unused)

### cell-11 — Ingredient model
- **Before:** Load `all-MiniLM-L6-v2` via SentenceTransformer (384-dim)
- **After:** Delete cell entirely. CLIP text encoder (already loaded) handles ingredients.

### cell-16 — Ingredient embeddings
- **Before:** `ingredient_model.encode(df["ingredients_clean"])` → 384-dim
- **After:** `get_clip_text_embeddings(df["ingredients_clean"])` → 512-dim (consistent CLIP space)

### cell-17 — Zero-padding
- **Before:** Pads 384→512 with zeros, renormalizes (mathematically lossy hack)
- **After:** Delete cell entirely. No padding needed.

### cell-18 — Fusion weight
- **Before:** `alpha = 0.7`
- **After:** `alpha = 0.9` (ablation shows best R@1=21.6%, R@10=59.2%)

### cell-27 — Evaluation
- **Before:** Recall@K only (K=1,5,10)
- **After:** Add median rank (medR) per direction. Lower = better.

---

## New Additions

### medR metric function
```python
def median_rank(query_embs, index, n_total):
    scores, indices = index.search(query_embs.astype("float32"), n_total)
    ranks = [int(np.where(indices[i] == i)[0][0]) + 1 for i in range(len(query_embs))]
    return float(np.median(ranks))
```
Added to eval section. Results table gains `medR` column.

### Interactive demo cell (final cell)
- User pastes any food image URL
- Cell fetches image, displays it, runs through CLIP vision encoder, queries FAISS index
- Prints top-5 recipes with title, similarity score, first 200 chars of ingredients
- Single variable (`url`) to change — no other edits needed

---

## What Does NOT Change

- Dataset: same 13,471 → 3,000 cap
- CLIP model: `openai/clip-vit-base-patch32` (unchanged)
- FAISS index type: `IndexFlatIP` (exact search, fine at 3k scale)
- Evaluation protocol: Recall@1/5/10 (medR added, not replacing)
- Notebook cell count: net -2 cells (remove cell-11 and cell-17), +2 new cells (medR helper, interactive demo)

---

## Expected Metric Improvement

| Metric | Before | After |
|--------|--------|-------|
| Im→Recipe R@1 | 19.9% | ~21.6% |
| Im→Recipe R@10 | 55.0% | ~59.2% |
| Re→Image R@10 | 59.1% | ~60%+ |
| medR (Im→Recipe) | ~N/A | added |
| Zero-padding hack | present | removed |
| GPU utilization | 0% (CPU forced) | T4 active |

---

## Dependencies

No new pip installs. `sentence-transformers` import removed.
All functionality uses `transformers`, `faiss-cpu`, `torch`, `PIL`, `requests`.
