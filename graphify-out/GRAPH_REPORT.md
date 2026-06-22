# Graph Report - ML_final_project  (2026-06-22)

## Corpus Check
- 51 files · ~7,590,700 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 337 nodes · 433 edges · 39 communities (29 shown, 10 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ab9a8b46`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]

## God Nodes (most connected - your core abstractions)
1. `load()` - 20 edges
2. `JointEmbeddingModel` - 18 edges
3. `InfoNCELoss` - 16 edges
4. `FusionModule` - 15 edges
5. `RecipeDataset` - 14 edges
6. `TextEncoder` - 14 edges
7. `main()` - 14 edges
8. `todo.md — Food Image-to-Recipe Retrieval with Ingredient-Aware Fusion` - 14 edges
9. `compute_metrics()` - 12 edges
10. `ImageEncoder` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_output_file_created()` --calls--> `Path`  [INFERRED]
  tests/test_phase2_precompute.py → src/training/train.py
- `test_getitem_image_feat_shape()` --calls--> `RecipeDataset`  [EXTRACTED]
  tests/test_phase2_dataset.py → src/data/build_dataset.py
- `test_getitem_keys()` --calls--> `RecipeDataset`  [EXTRACTED]
  tests/test_phase2_dataset.py → src/data/build_dataset.py
- `test_getitem_metadata_types()` --calls--> `RecipeDataset`  [EXTRACTED]
  tests/test_phase2_dataset.py → src/data/build_dataset.py
- `test_getitem_token_shapes()` --calls--> `RecipeDataset`  [EXTRACTED]
  tests/test_phase2_dataset.py → src/data/build_dataset.py

## Import Cycles
- None detected.

## Communities (39 total, 10 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (27): FusionModule, ImageEncoder, TextEncoder, Tensor, Tensor, DictConfig, Tensor, Tensor (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (26): _assign_splits(), load(), _parse_ingredients(), _parse_instructions(), Kaggle food-recipe dataset adapter.  Reads the Kaggle CSV (pes12017000148/food-i, Assign 'partition' in-place using deterministic index-based split., Load Kaggle dataset and return normalized recipe dicts.      Args:         cfg:, Parse ingredient field: Python list literal string → List[str]. (+18 more)

### Community 2 - "Community 2"
Cohesion: 0.19
Nodes (18): AdamW, DataLoader, JointEmbeddingModel, InfoNCELoss, JointEmbeddingModel, Path, Tensor, Tensor (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (18): main(), Precompute CLIP ViT-B-32 image features and save {stem: Tensor(512)} dict., DictConfig, DictConfig, Phase 0 verify: configs load, seed is reproducible, device resolves., test_baseline_config_loads_with_data_include(), test_cli_override_changes_resolved_config(), test_fusion_config_uses_attention() (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (21): Behavior, Class, CLI Usage, Config Fix, Constraints, Contract, Data Flow, Dataset (`src/data/build_dataset.py`) (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (14): get_split(), RecipeDataset: loads cached image features + tokenizes text on-the-fly., RecipeDataset, Dataset, DictConfig, Phase 2: RecipeDataset contract tests., test_get_split_train_only(), test_getitem_image_feat_shape() (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.13
Nodes (14): Locked decisions, Open items — RESOLVED, Phase 0 — Scaffold + env  ✅ DONE (commit 2c3dba1), Phase 1 — Data pipeline  (test-first), Phase 2 — Image feature precompute (Kaggle), Phase 3 — Models (shape tests first), Phase 4 — Losses (test-first), Phase 5 — Train loop (+6 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (12): Architecture, Constraints, Context, Eval Metrics — `src/eval/metrics.py`, File Map, Fusion Module — `src/models/fusion.py`, Image Tower — `src/models/image_encoder.py`, Joint Model — `src/models/joint_embedding.py` (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (11): 10. Definition of done (per task), 1. Project summary, 2. Commands (use these exact ones), 3. Tech stack (specific versions, no "just use latest"), 4. Project structure, 5. ML methodology rules (the part generic agents get wrong), 6. Reproducibility (non-negotiable), 7. Code style (+3 more)

### Community 9 - "Community 9"
Cohesion: 0.17
Nodes (11): File Map, Global Constraints, Phase 3: Model, Loss, Train Loop, Eval — Implementation Plan, Self-Review, Task 1: ImageEncoder, Task 2: TextEncoder, Task 3: FusionModule, Task 4: JointEmbeddingModel (+3 more)

### Community 10 - "Community 10"
Cohesion: 0.17
Nodes (11): Chosen: Use `Cleaned_Ingredients` when available, Decision 1: Feature Precomputation Strategy, Decision 2: Text Input Architecture, Decision 3: Cleaned_Ingredients vs Raw Ingredients, Option 1 (CHOSEN): Dict-keyed `.pt` cache, Option 2 (REJECTED): HDF5 memory-mapped file, Option 3 (REJECTED): On-the-fly CLIP inference (no precompute), Option A (CHOSEN): Two separate text streams (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (9): compute_metrics(), Compute medR and R@k for both retrieval directions.      Args:         image_emb, Tensor, test_custom_ks(), test_metrics_keys_present(), test_perfect_retrieval_medR_1(), test_perfect_retrieval_R_at_1_is_100(), test_r_at_k_between_0_and_100() (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.20
Nodes (9): Global Constraints, Phase 2: Precompute + Dataset Implementation Plan, Placeholder Scan, Self-Review, Spec Coverage, Task 1: Config Fix, Task 2: Precompute Script (TDD), Task 3: RecipeDataset (TDD) (+1 more)

### Community 13 - "Community 13"
Cohesion: 0.20
Nodes (9): Adapter Contract, Chosen Approach, Config Changes (`configs/data.yaml`), Constraints / Risks, Dataset Adaptation Design: Recipe1M → Kaggle Food Dataset, Internal Normalized Format, Phase / Todo Changes, Problem (+1 more)

### Community 14 - "Community 14"
Cohesion: 0.31
Nodes (7): _make_mock_clip(), Phase 2: precompute_image_feats.main() contract tests. CLIP is mocked., test_corrupt_image_skipped(), test_dict_keys_are_stems(), test_feat_shape_is_512(), test_no_images_raises(), test_output_file_created()

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (8): File Map, Phase 1: Kaggle Data Adapter Implementation Plan, Self-Review Checklist, Task 1: Update `configs/data.yaml`, Task 2: Create test fixture CSV, Task 3: Write failing tests for kaggle_adapter, Task 4: Implement `src/data/kaggle_adapter.py`, Task 5: Download note + path verification

### Community 16 - "Community 16"
Cohesion: 0.22
Nodes (8): Commit Hash, Fixes Applied, Full Suite, Phase 2 Tests, Status: DONE, Task 3 Report: RecipeDataset + Error/Logging Fixes, Test Results, Verification

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (7): Changes Made, Commit, Concerns, Status, Summary, Task 1 Report: Config Fix, Verification

### Community 18 - "Community 18"
Cohesion: 0.33
Nodes (5): Data, Environment / compute, lessons.md — Food Image-to-Recipe Retrieval, Modeling decisions (the parts generic agents get wrong), Process

### Community 19 - "Community 19"
Cohesion: 0.33
Nodes (5): Commits, Full Suite, Notes / Concerns, Task 2 Report: Precompute Image Features, Test Output

### Community 20 - "Community 20"
Cohesion: 0.40
Nodes (4): Design Decisions & Lessons, Phase 1: Dataset Choice, Phase 2: Feature Precomputation Strategy, Phase 2: Text Input Design

## Knowledge Gaps
- **126 isolated node(s):** `allow`, `DictConfig`, `Tensor`, `Tensor`, `Tensor` (+121 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Path` connect `Community 2` to `Community 1`, `Community 3`, `Community 5`, `Community 14`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `JointEmbeddingModel` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Why does `load()` connect `Community 1` to `Community 2`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `JointEmbeddingModel` (e.g. with `AdamW` and `DataLoader`) actually correct?**
  _`JointEmbeddingModel` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `InfoNCELoss` (e.g. with `AdamW` and `DataLoader`) actually correct?**
  _`InfoNCELoss` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `FusionModule` (e.g. with `JointEmbeddingModel` and `DictConfig`) actually correct?**
  _`FusionModule` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `allow`, `RecipeDataset: loads cached image features + tokenizes text on-the-fly.`, `Kaggle food-recipe dataset adapter.  Reads the Kaggle CSV (pes12017000148/food-i` to the rest of the system?**
  _146 weakly-connected nodes found - possible documentation gaps or missing edges._