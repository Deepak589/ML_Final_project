# Phase 2 Design Options Analysis

All options considered during Phase 2 brainstorming, with pros/cons and final recommendation rationale.

---

## Decision 1: Feature Precomputation Strategy

### Option 1 (CHOSEN): Dict-keyed `.pt` cache

Store CLIP features as `{image_stem: Tensor(512)}` dict, saved with `torch.save()` to `data/processed/image_feats.pt`. Dataset loads the full dict once at `__init__`.

**Pros:**
- Zero extra dependencies — `torch.save/load` already in the stack
- 13k × 512 × float32 = ~26MB — loads into RAM in milliseconds
- O(1) lookup by image stem string key
- Easy to inspect/debug: `torch.load("image_feats.pt")["recipe-slug"]`
- Idempotent: rerunning precompute script just overwrites the file
- Works with any device (CPU/MPS/CUDA) — just a dict of tensors

**Cons:**
- Loads entire dict into RAM (fine at 26MB, would be a problem at 1M+ images)
- Not suitable if features need partial updates (must recompute all)

**Why it wins:** Right-sized for the dataset. Simple to implement, simple to debug, zero complexity overhead. The "cons" only matter at a scale 100× larger than this dataset.

---

### Option 2 (REJECTED): HDF5 memory-mapped file

Store features in an HDF5 file using `h5py`. Dataset memory-maps the file — only the rows that are accessed are loaded into RAM.

**Pros:**
- Memory-efficient at scale: 1M+ images, only loads what's needed
- Standard in large-scale ML pipelines (e.g., full Recipe1M)
- Supports partial updates (overwrite specific rows)
- Supports concurrent reads from multiple DataLoader workers

**Cons:**
- Requires `h5py` dependency (not currently in requirements)
- More complex Dataset code (integer index → HDF5 row mapping)
- HDF5 doesn't natively support string keys — need a separate index mapping file
- Overkill for 26MB of data — memory-mapping adds latency for small files
- Harder to inspect/debug than a plain dict

**Why rejected:** Zero benefit at 13k scale. Adds a dependency and complexity purely for the sake of "production patterns." Would be the right choice at Recipe1M scale (1M images, ~2GB of features).

---

### Option 3 (REJECTED): On-the-fly CLIP inference (no precompute)

Skip precomputation entirely. Run CLIP ViT-B-32 inside the DataLoader `__getitem__`, compute image features during training.

**Pros:**
- No separate precompute step — simpler pipeline
- No disk space for cached features
- Works correctly even if images change (always fresh)

**Cons:**
- CLIP ViT-B-32 inference: ~200ms per batch on CPU, ~20ms on GPU
- Over 30 epochs × 160 batches = 4,800 forward passes through CLIP
- CPU: 4,800 × 200ms = ~960 seconds (~16 min) of pure CLIP overhead
- CLIP weights are FROZEN — outputs never change during training, so recomputing is pure waste
- DataLoader workers can't share GPU — CLIP would run on CPU in workers, making it even slower
- Training becomes I/O + inference bound instead of model-bound

**Why rejected:** Since CLIP is frozen (not fine-tuned), its outputs are identical every epoch. Recomputing them 30 times is logically equivalent to redoing the same arithmetic 30 times. Precomputing once is strictly better in every way.

---

## Decision 2: Text Input Architecture

### Option A (CHOSEN): Two separate text streams

Dataset returns `ingr_input_ids` and `instr_input_ids` as separate tensors. DistilBERT encoder runs twice per sample in the model forward pass — once for ingredients, once for instructions.

**Pros:**
- Enables `fusion.mode: ingr_only` ablation (already in config) — test whether ingredients alone are sufficient for retrieval
- Matches original im2recipe architecture (separate ingredient/instruction encoders)
- Cleaner gradient flow — each stream's loss gradient flows back independently
- Easier to analyze: can compare which stream contributes more to retrieval quality

**Cons:**
- Two forward passes through DistilBERT per sample (2× compute)
- Slightly more complex Dataset `__getitem__` (4 tensors instead of 2)
- `share_encoder: true` means the same weights see both — some argue a single combined input is more natural

**Why it wins:** The ablation (`ingr_only` mode) is already planned and requires separate streams. Building concatenated text would force a refactor later. Two forward passes at DistilBERT-base scale (~66M params) is fast — not a meaningful bottleneck.

---

### Option B (REJECTED): Concatenated text (ingredients + [SEP] + instructions)

Join ingredients and instructions into a single string with `[SEP]` separator. One DistilBERT forward pass per sample.

**Pros:**
- Single forward pass — 2× faster text encoding
- Simpler Dataset code (2 tensors instead of 4)
- DistilBERT was pre-trained on full documents — a longer combined input may capture cross-ingredient-instruction context better

**Cons:**
- Loses `ingr_only` ablation capability entirely — can't separate ingredient signal from instruction signal
- Combined max length = 24 + 128 = 152 tokens, which exceeds DistilBERT's typical usage (128 default) — need to handle truncation carefully
- Cannot independently scale `ingr_max_tokens` vs `instr_max_tokens` in config
- Standard im2recipe baselines use separate streams — deviating makes it harder to explain results relative to the literature

**Why rejected:** The `ingr_only` ablation is already in `configs/baseline.yaml`. Concatenating text makes that ablation impossible without refactoring. The 2× compute cost of separate streams is negligible at this scale.

---

## Decision 3: Cleaned_Ingredients vs Raw Ingredients

### Chosen: Use `Cleaned_Ingredients` when available

The Kaggle CSV has both `Ingredients` (raw, e.g., `"['1 cup flour', '2 eggs, beaten']"`) and `Cleaned_Ingredients` (pre-parsed, e.g., `"['flour', 'eggs']"`).

**Why `Cleaned_Ingredients`:**
- Strips quantities and units — model learns ingredient identity, not "1 cup" vs "2 cups"
- Reduces token noise: "1 tablespoon extra-virgin olive oil" → "olive oil"
- Shorter sequences → more ingredients fit within `ingr_max_tokens: 24`
- Quantities are not useful for cross-modal retrieval (images don't show measurements)

**Why not raw `Ingredients`:**
- "1 cup flour, sifted" and "flour" mean the same thing visually
- Wasted tokens on quantities/units push actual ingredient names out of the 24-token window

---

## Summary Table

| Decision | Chosen | Key Reason |
|---|---|---|
| Cache format | Dict `.pt` | Right-sized (26MB), zero extra deps |
| Precompute vs on-the-fly | Precompute | CLIP frozen → recomputing 30× is pure waste |
| Text streams | Separate ingr + instr | Enables `ingr_only` ablation already in config |
| Ingredients column | `Cleaned_Ingredients` | Less token noise, more ingredients fit in window |
