# Design Decisions & Lessons

A record of architectural choices with rejected alternatives — useful for project presentations and explaining "why not X".

---

## Phase 2: Feature Precomputation Strategy

**Decision: Dict-keyed `.pt` cache (Option 1)**

Store precomputed CLIP image features as a Python dict `{image_stem: Tensor(512)}` saved to `data/processed/image_feats.pt`. Dataset loads the full dict at `__init__`, lookups are O(1) by key.

**Why this works:**
- 13k images × 512-dim × float32 ≈ 26MB — fits in RAM trivially
- `.pt` / `torch.load` is zero-dependency (already using PyTorch)
- Dict lookup is constant time — no index management needed
- Simple to inspect: `feats = torch.load("image_feats.pt"); feats["recipe-slug"]`

**Rejected: HDF5 memory-mapped file (Option 2)**
HDF5 with `h5py` memory-maps data so only accessed rows are loaded. Standard at million-scale datasets (e.g., full Recipe1M with 1M images). For 26MB it adds a dependency (`h5py`), more complex Dataset code, and zero practical benefit. Would be the right call at 1M+ images.

**Rejected: On-the-fly CLIP inference (Option 3)**
Run CLIP ViT-B-32 during training DataLoader workers instead of caching. Keeps code clean (no separate precompute step) but CLIP inference adds ~200ms/batch. Over 30 epochs × ~160 batches = ~960 extra seconds per run (~16 min). Since CLIP weights are frozen (not fine-tuned), outputs never change — there is no reason to recompute them.

---

## Phase 2: Text Input Design

**Decision: Two separate text streams (ingredients + instructions)**

Dataset returns `ingr_input_ids`, `ingr_attention_mask`, `instr_input_ids`, `instr_attention_mask` as separate tensors. DistilBERT encoder runs twice per sample.

**Why:**
- Config has `fusion.mode: ingr_only` ablation option — requires ingredients to be accessible independently
- Matches standard im2recipe architecture (separate ingredient and instruction encoders)
- Allows future ablation: "does adding instructions help over ingredients alone?"

**Rejected: Concatenated text (ingredients + [SEP] + instructions)**
Single forward pass, simpler code. But loses ingredient/instruction separation, making the `ingr_only` ablation impossible without a refactor. Since the ablation is already planned, the cost of this simplification is paid later.

---

## Phase 1: Dataset Choice

**Decision: Kaggle `pes12017000148/food-ingredients-and-recipe-dataset-with-images`**

~13k paired (image, recipe) samples. Replaced Recipe1M (original plan) after Recipe1M registration broke.

**Why acceptable for course project:**
- Paired image + title + ingredients + instructions — same structure as Recipe1M
- 13k samples sufficient to train and evaluate cross-modal retrieval (1k eval subset)
- No paper comparability required for course demonstration

**Tradeoff vs Recipe1M:**
- Recipe1M has 1M recipes — better generalization, standard benchmark
- 13k means lower absolute Recall@K scores (retrieval from 1k vs 10k pool)
- Eval subset capped at 1k (not 1k + 10k as in published baselines)
