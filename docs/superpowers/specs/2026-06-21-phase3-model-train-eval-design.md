# Phase 3: Model, Loss, Train Loop, Eval — Design Spec

Date: 2026-06-21

## Context

Data pipeline (Phases 1–2) complete. RecipeDataset yields 7-key dicts per sample:
`image_feat (512)`, `ingr_input_ids`, `ingr_attention_mask`, `instr_input_ids`,
`instr_attention_mask`, `recipe_id`, `partition`.

Dataset: 13,471 valid image-recipe pairs (Kaggle Food Ingredients + Image Name Mapping CSV).
No category labels → `semantic.enabled` must be `false`; semantic aux loss dropped.

---

## Architecture

### Image Tower — `src/models/image_encoder.py`

```
image_feat (B, 512)
  → Linear(512, proj_hidden=1024) + LayerNorm + GELU
  → Linear(1024, embed_dim=1024) + LayerNorm
  → L2-normalize
→ image_emb (B, 1024)
```

Config keys: `model.image.in_dim`, `model.image.proj_hidden`, `model.embed_dim`.

### Text Tower — `src/models/text_encoder.py`

Single shared DistilBERT (`distilbert-base-uncased`) processes both ingredient and
instruction streams independently. **Pooling: masked mean** over non-padding tokens
(permutation-robust for ingredient lists, solid for instructions).

```
ingr_input_ids, ingr_attention_mask
  → shared DistilBERT → last_hidden_state (B, L, 768)
  → masked mean pool → (B, 768)
  → ingr_proj: Linear(768, embed_dim) + LayerNorm
→ ingr_emb (B, 1024)

instr_input_ids, instr_attention_mask
  → shared DistilBERT → last_hidden_state (B, L, 768)
  → masked mean pool → (B, 768)
  → instr_proj: Linear(768, embed_dim) + LayerNorm
→ instr_emb (B, 1024)
```

Config keys: `model.text.encoder`, `model.text.hidden`, `model.text.freeze_text`,
`model.text.share_encoder` (always true here).

### Fusion Module — `src/models/fusion.py`

Three modes from `model.fusion.mode`:

**concat** (baseline):
```
cat(ingr_emb, instr_emb) → (B, 2048)
  → Linear(2048, fusion_hidden=1024) + GELU + Linear(1024, embed_dim) + LayerNorm
  → L2-normalize
→ recipe_emb (B, 1024)
```

**attention** (fusion config):
```
# instr queries attend over ingr token sequence
ingr_tokens: (B, L_i, 768) from DistilBERT (before pooling)
instr_emb:   (B, 1024) as query
MultiheadAttention(embed_dim=1024, n_heads=8)
  query = instr_emb.unsqueeze(1)   # (B, 1, 1024)
  key = value = ingr_proj_tokens    # (B, L_i, 1024) via linear
  → attended (B, 1, 1024).squeeze(1)
  → add instr_emb (residual) + LayerNorm
  → L2-normalize
→ recipe_emb (B, 1024)
```

**ingr_only** (ablation):
```
ingr_emb → L2-normalize → recipe_emb
```

### Joint Model — `src/models/joint_embedding.py`

Top-level module wiring all towers. `forward(batch) → (image_emb, recipe_emb)`.
Both outputs are L2-normalized (1024-d). Handles `freeze_text` flag by toggling
`requires_grad` on DistilBERT params after init.

---

## Loss — `src/losses/infonce.py`

Symmetric InfoNCE (NT-Xent) over in-batch negatives:

```python
logits_i2r = image_emb @ recipe_emb.T / temperature   # (B, B)
logits_r2i = recipe_emb @ image_emb.T / temperature   # (B, B)
labels = torch.arange(B)
loss = 0.5 * (CE(logits_i2r, labels) + CE(logits_r2i, labels))
```

Temperature: `loss.temperature = 0.07` (fixed, from config). No learnable temp for now.
Semantic loss: skipped entirely (no category labels in dataset).

---

## Train Loop — `src/training/train.py`

Entry point: `python -m src.training.train --config baseline.yaml [dot.overrides...]`

```
load_config(args.config, overrides)
set_seed(cfg.seed)
device = resolve_device(cfg.device)
train_loader, val_loader = get_split(cfg, "train"), get_split(cfg, "val")
model = JointEmbeddingModel(cfg).to(device)
optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
scaler = GradScaler(enabled=cfg.train.amp)

for epoch in range(cfg.train.epochs):
    # train
    model.train()
    for step, batch in enumerate(train_loader):
        with autocast(enabled=cfg.train.amp):
            image_emb, recipe_emb = model(batch)
            loss = criterion(image_emb, recipe_emb) / cfg.train.grad_accum
        scaler.scale(loss).backward()
        if (step+1) % cfg.train.grad_accum == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        tb_writer.add_scalar("train/loss", loss, global_step)

    # val
    model.eval()
    metrics = evaluate(model, val_loader, cfg, device)
    tb_writer.add_scalars("val", metrics, epoch)
    if metrics["im2recipe_medR"] < best_medR:
        best_medR = metrics["im2recipe_medR"]
        save_checkpoint(model, optimizer, epoch, cfg)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= cfg.train.early_stop_patience:
            break
```

Checkpoint saved to `runs/{exp_name}/best.pt` — contains model state dict + epoch + val medR.

---

## Eval Metrics — `src/eval/metrics.py`

```python
def compute_metrics(image_embs, recipe_embs, ks=(1, 5, 10)):
    # image_embs, recipe_embs: (N, D) L2-normalized tensors
    # returns dict with medR, R@1, R@5, R@10 for both directions
    sim = image_embs @ recipe_embs.T   # (N, N) cosine sim (already L2-normed)
    # im2recipe: for each image, rank recipes
    # recipe2im: transpose sim, for each recipe rank images
    ...
```

1k subset eval: sample 1000 from val/test, repeat `n_folds=10` → mean metrics.
Headline numbers: mean ± std over 3 seeds (per lessons.md rule).

Metrics reported:
- `im2recipe_medR`, `im2recipe_R@1`, `im2recipe_R@5`, `im2recipe_R@10`
- `recipe2im_medR`, `recipe2im_R@1`, `recipe2im_R@5`, `recipe2im_R@10`

---

## File Map

| File | Class/Function | Purpose |
|------|---------------|---------|
| `src/models/image_encoder.py` | `ImageEncoder` | 512→1024 projection + L2-norm |
| `src/models/text_encoder.py` | `TextEncoder` | Shared DistilBERT + masked mean pool + 2 proj heads |
| `src/models/fusion.py` | `FusionModule` | concat \| attention \| ingr_only |
| `src/models/joint_embedding.py` | `JointEmbeddingModel` | Wires towers, forward→(image_emb, recipe_emb) |
| `src/losses/infonce.py` | `InfoNCELoss` | Symmetric NT-Xent, fixed temp |
| `src/training/train.py` | `main()` | Train loop, AMP, early stop, TensorBoard, ckpt |
| `src/eval/metrics.py` | `compute_metrics()` | medR + R@k both directions |
| `tests/test_phase3_models.py` | — | Shape tests for all modules |
| `tests/test_phase3_loss.py` | — | InfoNCE correctness (diagonal = min loss) |
| `tests/test_phase3_metrics.py` | — | medR=1 when sim is identity |

---

## Constraints

- MPS-safe: no `torch.cuda.*` calls in model code; use `cfg.device` throughout
- Ingredient order is NOT semantically meaningful → masked mean pool (not positional)
- DistilBERT caps at 512 tokens (sufficient for both streams)
- `faiss-cpu` for any ANN search if added later; no `faiss-gpu`
- Never recompute CLIP image feats per run — always load from `image_feats.pt`
- Eval must be deterministic given (ckpt, subset, seed)
