# todo.md — Food Image-to-Recipe Retrieval with Ingredient-Aware Fusion

> Project plan. Each item has a **verify** check. Mark `[x]` when verified, not when written.
> Build order is dependency-ordered. Don't skip ahead.

## Locked decisions

| Decision | Value | Why |
|---|---|---|
| Image encoder | CLIP ViT-B/32, **frozen**, feats **cached** | expensive part; cache once → ~10x faster |
| Image dim | 512-d | CLIP ViT-B/32 output |
| Recipe text encoder | DistilBERT (768-d, 512 tokens) | CLIP text caps at 77 tokens; recipes too long |
| Text encoder trainable? | **Yes, light fine-tune** (config `freeze_text` to ablate) | fusion needs token-level streams; only light tower trains |
| Shared embed dim D | 1024, L2-normalized, cosine | AGENTS.md |
| Default loss | InfoNCE symmetric, in-batch negs (triplet = switch) | strong + simple at our scale |
| Fusion | attention (ablate: concat \| attention \| ingr-only) | the project contribution |
| Semantic-reg aux | **ON v1**: coarse title-keyword category labels, `lambda_sem` config | baseline improves medR with it |
| Subset size N | **2k smoke first → then 50k** (config `n_recipes`) | fast first loop, then scale |
| Tracking | **TensorBoard** (local, offline) | no account/internet needed |
| Eval | FAISS-cpu, 1k + 10k, both directions, 10 random folds, fixed seed | im2recipe-standard |
| Dev vs train | Mac = code+debug; Kaggle = precompute+train+eval | see lessons.md |

## Open items — RESOLVED
- [x] Subset N: 2k smoke → 50k real
- [x] Tracking: TensorBoard
- [x] Semantic categories: ON v1, coarse title-keyword labels

---

## Phase 0 — Scaffold + env
1. [ ] Create dir tree: `configs/ src/{data,models,losses,utils} tests/ runs/ notebooks/` → verify: `tree` matches AGENTS.md §4
2. [ ] `.gitignore` (data/, runs/, *.pt, __pycache__, .DS_Store) → verify: `git status` ignores them
3. [ ] `requirements.txt` pinned (torch, torchvision, transformers, faiss-cpu, omegaconf, ftfy/clip or open_clip, h5py/lmdb, pytest, ruff) → verify: `pip install -r` clean on Mac
4. [ ] `configs/{data,baseline,fusion}.yaml` skeletons (OmegaConf) → verify: load + print resolved
5. [ ] `src/utils/seed.py` (python/numpy/torch/cudnn) + `git init`, first commit → verify: `pytest -q` collects, seed sets reproducibly

## Phase 1 — Data pipeline  (test-first)
6. [ ] Tiny fixtures: 8 fake recipes + 8 random 512-d "image feats" in `tests/fixtures/` → verify: loadable
7. [ ] `src/data/build_dataset.py`: read layer1/layer2/det_ingrs, keep recipes with ≥1 local image, sample N, use `partition` split (no id leak), write index (parquet/pkl) → verify: test asserts zero id overlap across splits
8. [ ] `src/data/dataset.py` `Recipe1MDataset`: returns (img_feat, ingr_text, instr_text, id, sem_label) + collate that tokenizes batch → verify: shapes test on fixtures
9. [ ] `src/data/tokenizers.py`: DistilBERT tokenizer wrap, ingr vs instr formatting, len caps (ingr≤24, instr≤128) → verify: cap respected, padding mask correct

## Phase 2 — Image feature precompute (Kaggle)
10. [ ] `src/data/precompute_image_feats.py --split X`: CLIP ViT-B/32 over images, store fp16 keyed by id (HDF5/LMDB) → verify: count == #samples, dim 512, deterministic
11. [ ] Smoke on Mac with ~50 images → verify: file written, re-load matches

## Phase 3 — Models (shape tests first)
12. [ ] `src/models/image_encoder.py`: img_proj MLP 512→1024 + L2norm → verify: (B,512)->(B,1024), ‖·‖≈1
13. [ ] `src/models/text_encoder.py`: shared DistilBERT, returns last_hidden_state + mask → verify: (B,L,768)
14. [ ] `src/models/{ingredient,instruction}_encoder.py`: stream heads + permutation-robust pool → verify: ingredient output invariant to shuffle (mean/attn pool)
15. [ ] `src/models/fusion.py`: `mode ∈ {concat, attention, ingr_only}` common interface → verify: all 3 modes return (B,Dh)
16. [ ] `src/models/joint_model.py`: image tower + recipe tower(text→fusion→recipe_proj) + optional sem head → verify: forward returns img_emb,recipe_emb (B,1024) normalized, fixture batch

## Phase 4 — Losses (test-first)
17. [ ] `src/losses/infonce.py` symmetric, temp τ → verify: loss↓ when embeddings aligned; gradient finite
18. [ ] `src/losses/triplet.py` + hard/semi-hard mining → verify: zero loss when margin satisfied
19. [ ] `src/losses/semantic_reg.py` aux CE, λ weight → verify: combines without NaN

## Phase 5 — Train loop
20. [ ] `src/train.py`: AMP, grad-accum, ckpt+resume, early-stop on val medR, TB/W&B logging, writes `runs/<exp>/config_used.yaml` + git SHA → verify: 1-epoch overfit on 8-sample fixture drives loss→~0
21. [ ] CLI `--config` only; flags override keys → verify: override changes resolved config

## Phase 6 — Eval
22. [ ] `src/utils/metrics.py`: medR, R@{1,5,10} → verify: unit test on hand-built ranking (known answer)
23. [ ] `src/eval.py --ckpt --subset {1k,10k} --direction both`: FAISS index, 10 folds, fixed seed, mean medR → verify: deterministic across reruns; runs end-to-end on fixture
24. [ ] re-run produces identical numbers → verify: byte-equal metric output

## Phase 7 — Baseline run (Kaggle)
25. [ ] Precompute feats for subset on Kaggle → verify: feat files for train/val/test
26. [ ] Train no-fusion / concat baseline → verify: val medR improves vs random (random medR≈N/2)
27. [ ] Eval 1k + 10k both directions → record in experiment log → verify: numbers logged with seed+subset

## Phase 8 — Fusion + ablation
28. [ ] Train attention fusion → verify: trains, ckpt saved
29. [ ] Train ingr-only → verify: trains
30. [ ] Ablation table: concat vs attention vs ingr-only, 1k+10k, both dirs → verify: table in experiment log, attention ≥ concat (hypothesis)

## Phase 9 — Reproducibility + writeup
31. [ ] Re-run best config × 3 seeds → verify: report mean±std
32. [ ] Experiment log `runs/EXPERIMENTS.md` complete → verify: every headline has seed count + subset
33. [ ] `pytest -q` + `ruff check . && ruff format --check .` clean → verify: green

---

## Review (fill when done)
- (pending)
