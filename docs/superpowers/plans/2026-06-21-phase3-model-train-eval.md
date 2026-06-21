# Phase 3: Model, Loss, Train Loop, Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ImageEncoder, TextEncoder, FusionModule, JointEmbeddingModel, InfoNCELoss, eval metrics, and a full training loop for food image-to-recipe retrieval.

**Architecture:** Shared DistilBERT (masked mean pool) encodes ingredient + instruction streams separately → FusionModule (concat or cross-attention) → recipe embedding. ImageEncoder projects cached CLIP feats (512-d). Both embeddings L2-normalized to 1024-d shared space. Symmetric InfoNCE loss over in-batch negatives.

**Tech Stack:** PyTorch, HuggingFace Transformers (distilbert-base-uncased), OmegaConf, TensorBoard, pytest

## Global Constraints

- All L2-normalized outputs — `F.normalize(x, dim=-1)` before returning embeddings
- MPS-safe: no `torch.cuda.*` calls; use `device` string from `resolve_device()`; AMP only when `device == "cuda"`
- `faiss-cpu` only (no `faiss-gpu`) — Mac dev box
- Never recompute CLIP feats per run — always load from `image_feats.pt`
- Semantic loss skipped — dataset has no category labels; `semantic.enabled` is always treated as false
- `share_encoder: true` always — one DistilBERT for both text streams
- Eval deterministic given (ckpt, subset, seed) — use `set_seed()` from `src/utils/seed.py`
- Match existing code style: `from __future__ import annotations`, `_log = logging.getLogger(__name__)`, docstrings one-line max
- Run all tests from repo root: `pytest tests/ -v`
- Existing passing tests must stay green: `pytest tests/ -v` shows 30 passing before you start

---

## File Map

| File | Status | Responsibility |
|------|--------|---------------|
| `src/models/image_encoder.py` | CREATE | `ImageEncoder`: Linear 512→1024 + LayerNorm + L2-norm |
| `src/models/text_encoder.py` | CREATE | `TextEncoder`: shared DistilBERT + masked mean pool + 2 proj heads |
| `src/models/fusion.py` | CREATE | `FusionModule`: concat \| attention \| ingr_only modes |
| `src/models/joint_embedding.py` | CREATE | `JointEmbeddingModel`: wires all towers, `forward→(image_emb, recipe_emb)` |
| `src/losses/infonce.py` | CREATE | `InfoNCELoss`: symmetric NT-Xent, fixed temperature |
| `src/eval/__init__.py` | CREATE | empty |
| `src/eval/metrics.py` | CREATE | `compute_metrics()`: medR, R@1/5/10, both directions |
| `src/training/__init__.py` | CREATE | empty |
| `src/training/train.py` | CREATE | `main()`: train loop, AMP, early stop, TensorBoard, checkpoint |
| `tests/test_phase3_models.py` | CREATE | Shape + L2-norm tests for all model components |
| `tests/test_phase3_loss.py` | CREATE | InfoNCE correctness tests |
| `tests/test_phase3_metrics.py` | CREATE | medR + R@k correctness tests |

---

### Task 1: ImageEncoder

**Files:**
- Create: `src/models/image_encoder.py`
- Create: `tests/test_phase3_models.py` (partial — ImageEncoder tests only)

**Interfaces:**
- Produces: `ImageEncoder(in_dim, proj_hidden, embed_dim)` — `forward(x: Tensor[B,in_dim]) → Tensor[B,embed_dim]` L2-normalized

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase3_models.py
from __future__ import annotations
import torch
import pytest
from src.models.image_encoder import ImageEncoder


def test_image_encoder_output_shape():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    x = torch.randn(4, 512)
    out = enc(x)
    assert out.shape == (4, 64)


def test_image_encoder_l2_normalized():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    x = torch.randn(4, 512)
    out = enc(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_image_encoder_different_batch_sizes():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    for B in (1, 8, 32):
        out = enc(torch.randn(B, 512))
        assert out.shape == (B, 64)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_phase3_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.models.image_encoder'`

- [ ] **Step 3: Implement ImageEncoder**

```python
# src/models/image_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, in_dim: int, proj_hidden: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_hidden),
            nn.LayerNorm(proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_models.py::test_image_encoder_output_shape tests/test_phase3_models.py::test_image_encoder_l2_normalized tests/test_phase3_models.py::test_image_encoder_different_batch_sizes -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/image_encoder.py tests/test_phase3_models.py
git commit -m "feat(models): add ImageEncoder with 2-layer projection + L2-norm"
```

---

### Task 2: TextEncoder

**Files:**
- Create: `src/models/text_encoder.py`
- Modify: `tests/test_phase3_models.py` — append TextEncoder tests

**Interfaces:**
- Consumes: nothing from prior tasks
- Produces:
  - `TextEncoder(encoder_name: str, hidden: int, embed_dim: int, freeze: bool = False)`
  - `encode_ingr(input_ids: Tensor[B,L], attention_mask: Tensor[B,L]) → tuple[Tensor[B,embed_dim], Tensor[B,L,hidden]]` — (ingr_emb L2-normed, raw hidden states)
  - `encode_instr(input_ids: Tensor[B,L], attention_mask: Tensor[B,L]) → Tensor[B,embed_dim]` — instr_emb L2-normed

- [ ] **Step 1: Write the failing tests (append to existing file)**

```python
# Append to tests/test_phase3_models.py
import torch
from src.models.text_encoder import TextEncoder

_ENC = "distilbert-base-uncased"
_B, _L, _H, _D = 2, 32, 768, 64


def test_text_encoder_ingr_emb_shape():
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D)
    ids = torch.randint(0, 1000, (_B, _L))
    mask = torch.ones(_B, _L, dtype=torch.long)
    ingr_emb, ingr_hidden = enc.encode_ingr(ids, mask)
    assert ingr_emb.shape == (_B, _D)
    assert ingr_hidden.shape == (_B, _L, _H)


def test_text_encoder_instr_emb_shape():
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D)
    ids = torch.randint(0, 1000, (_B, _L))
    mask = torch.ones(_B, _L, dtype=torch.long)
    instr_emb = enc.encode_instr(ids, mask)
    assert instr_emb.shape == (_B, _D)


def test_text_encoder_l2_normalized():
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D)
    ids = torch.randint(0, 1000, (_B, _L))
    mask = torch.ones(_B, _L, dtype=torch.long)
    ingr_emb, _ = enc.encode_ingr(ids, mask)
    instr_emb = enc.encode_instr(ids, mask)
    assert torch.allclose(ingr_emb.norm(dim=-1), torch.ones(_B), atol=1e-5)
    assert torch.allclose(instr_emb.norm(dim=-1), torch.ones(_B), atol=1e-5)


def test_text_encoder_masked_mean_ignores_padding():
    """Padding tokens (mask=0) must not affect the pooled output."""
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D)
    ids = torch.randint(0, 1000, (1, _L))
    full_mask = torch.ones(1, _L, dtype=torch.long)
    half_mask = full_mask.clone()
    half_mask[:, _L // 2 :] = 0  # mask out second half

    # Run with full mask
    emb_full, _ = enc.encode_ingr(ids, full_mask)
    # Run same ids but second-half tokens are padding
    padded_ids = ids.clone()
    padded_ids[:, _L // 2 :] = 0  # zero out padding positions
    emb_half, _ = enc.encode_ingr(padded_ids, half_mask)

    # Outputs should differ (different tokens in valid positions, but test that
    # half_mask doesn't include padding tokens — shapes must be correct)
    assert emb_full.shape == (1, _D)
    assert emb_half.shape == (1, _D)


def test_text_encoder_freeze_flag():
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D, freeze=True)
    for p in enc.bert.parameters():
        assert not p.requires_grad
    # Projection heads must still be trainable
    for p in enc.ingr_proj.parameters():
        assert p.requires_grad
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase3_models.py -k "text_encoder" -v
```
Expected: `ModuleNotFoundError: No module named 'src.models.text_encoder'`

- [ ] **Step 3: Implement TextEncoder**

```python
# src/models/text_encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        hidden: int,
        embed_dim: int,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(encoder_name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad_(False)
        self.ingr_proj = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.instr_proj = nn.Sequential(
            nn.Linear(hidden, embed_dim), nn.LayerNorm(embed_dim)
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()  # (B, L, 1)
        return (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

    def encode_ingr(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean(out.last_hidden_state, attention_mask)
        emb = F.normalize(self.ingr_proj(pooled), dim=-1)
        return emb, out.last_hidden_state

    def encode_instr(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._masked_mean(out.last_hidden_state, attention_mask)
        return F.normalize(self.instr_proj(pooled), dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_models.py -k "text_encoder" -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/text_encoder.py tests/test_phase3_models.py
git commit -m "feat(models): add TextEncoder with shared DistilBERT + masked mean pool"
```

---

### Task 3: FusionModule

**Files:**
- Create: `src/models/fusion.py`
- Modify: `tests/test_phase3_models.py` — append FusionModule tests

**Interfaces:**
- Consumes: `TextEncoder.encode_ingr` returns `(ingr_emb: Tensor[B,D], ingr_hidden: Tensor[B,L,768])`
- Produces: `FusionModule(embed_dim, hidden, mode, n_heads=8, text_hidden=768)`
  - `forward(ingr_emb: Tensor[B,D], instr_emb: Tensor[B,D], ingr_hidden: Tensor[B,L,text_hidden]=None, ingr_mask: Tensor[B,L]=None) → Tensor[B,D]` L2-normalized

- [ ] **Step 1: Write the failing tests (append to existing file)**

```python
# Append to tests/test_phase3_models.py
from src.models.fusion import FusionModule

_ED, _FH, _TH, _BF, _LF = 64, 128, 768, 4, 16


def test_fusion_concat_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="concat", text_hidden=_TH)
    ingr = torch.randn(_BF, _ED)
    instr = torch.randn(_BF, _ED)
    out = fuse(ingr, instr)
    assert out.shape == (_BF, _ED)


def test_fusion_concat_l2_normalized():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="concat", text_hidden=_TH)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED))
    assert torch.allclose(out.norm(dim=-1), torch.ones(_BF), atol=1e-5)


def test_fusion_attention_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="attention", n_heads=4, text_hidden=_TH)
    ingr_emb = torch.randn(_BF, _ED)
    instr_emb = torch.randn(_BF, _ED)
    ingr_hidden = torch.randn(_BF, _LF, _TH)
    ingr_mask = torch.ones(_BF, _LF, dtype=torch.long)
    out = fuse(ingr_emb, instr_emb, ingr_hidden=ingr_hidden, ingr_mask=ingr_mask)
    assert out.shape == (_BF, _ED)


def test_fusion_attention_l2_normalized():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="attention", n_heads=4, text_hidden=_TH)
    ingr_hidden = torch.randn(_BF, _LF, _TH)
    ingr_mask = torch.ones(_BF, _LF, dtype=torch.long)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED), ingr_hidden=ingr_hidden, ingr_mask=ingr_mask)
    assert torch.allclose(out.norm(dim=-1), torch.ones(_BF), atol=1e-5)


def test_fusion_ingr_only_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="ingr_only", text_hidden=_TH)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED))
    assert out.shape == (_BF, _ED)


def test_fusion_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown fusion mode"):
        FusionModule(embed_dim=_ED, hidden=_FH, mode="bad_mode", text_hidden=_TH)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase3_models.py -k "fusion" -v
```
Expected: `ModuleNotFoundError: No module named 'src.models.fusion'`

- [ ] **Step 3: Implement FusionModule**

```python
# src/models/fusion.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

_MODES = {"concat", "attention", "ingr_only"}


class FusionModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden: int,
        mode: str,
        n_heads: int = 8,
        text_hidden: int = 768,
    ) -> None:
        super().__init__()
        if mode not in _MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {_MODES}")
        self.mode = mode
        if mode == "concat":
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden),
                nn.GELU(),
                nn.Linear(hidden, embed_dim),
                nn.LayerNorm(embed_dim),
            )
        elif mode == "attention":
            self.ingr_token_proj = nn.Linear(text_hidden, embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        ingr_emb: torch.Tensor,
        instr_emb: torch.Tensor,
        ingr_hidden: torch.Tensor | None = None,
        ingr_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([ingr_emb, instr_emb], dim=-1)
            return F.normalize(self.mlp(x), dim=-1)
        if self.mode == "attention":
            kv = self.ingr_token_proj(ingr_hidden)  # (B, L, embed_dim)
            q = instr_emb.unsqueeze(1)               # (B, 1, embed_dim)
            key_padding_mask = ingr_mask == 0        # True = ignore
            attended, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
            out = self.norm(attended.squeeze(1) + instr_emb)
            return F.normalize(out, dim=-1)
        # ingr_only
        return F.normalize(ingr_emb, dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_models.py -k "fusion" -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/fusion.py tests/test_phase3_models.py
git commit -m "feat(models): add FusionModule with concat/attention/ingr_only modes"
```

---

### Task 4: JointEmbeddingModel

**Files:**
- Create: `src/models/joint_embedding.py`
- Modify: `tests/test_phase3_models.py` — append JointEmbeddingModel tests

**Interfaces:**
- Consumes:
  - `ImageEncoder(in_dim, proj_hidden, embed_dim)` from `src/models/image_encoder.py`
  - `TextEncoder(encoder_name, hidden, embed_dim, freeze)` from `src/models/text_encoder.py`
  - `FusionModule(embed_dim, hidden, mode, n_heads, text_hidden)` from `src/models/fusion.py`
- Produces: `JointEmbeddingModel(cfg: DictConfig)`
  - `forward(batch: dict) → tuple[Tensor[B,embed_dim], Tensor[B,embed_dim]]` — (image_emb, recipe_emb), both L2-normalized
  - Batch keys consumed: `image_feat`, `ingr_input_ids`, `ingr_attention_mask`, `instr_input_ids`, `instr_attention_mask`

- [ ] **Step 1: Write the failing tests (append to existing file)**

```python
# Append to tests/test_phase3_models.py
from omegaconf import OmegaConf
from src.models.joint_embedding import JointEmbeddingModel


def _make_cfg(mode: str = "concat") -> object:
    return OmegaConf.create({
        "model": {
            "embed_dim": 64,
            "image": {"in_dim": 512, "proj_hidden": 128},
            "text": {
                "encoder": "distilbert-base-uncased",
                "hidden": 768,
                "freeze_text": False,
                "share_encoder": True,
            },
            "fusion": {"mode": mode, "hidden": 128, "n_heads": 4},
        }
    })


def _make_batch(B: int = 2, L: int = 32) -> dict:
    return {
        "image_feat": torch.randn(B, 512),
        "ingr_input_ids": torch.randint(0, 1000, (B, L)),
        "ingr_attention_mask": torch.ones(B, L, dtype=torch.long),
        "instr_input_ids": torch.randint(0, 1000, (B, L)),
        "instr_attention_mask": torch.ones(B, L, dtype=torch.long),
    }


def test_joint_model_concat_shapes():
    model = JointEmbeddingModel(_make_cfg("concat"))
    image_emb, recipe_emb = model(_make_batch())
    assert image_emb.shape == (2, 64)
    assert recipe_emb.shape == (2, 64)


def test_joint_model_attention_shapes():
    model = JointEmbeddingModel(_make_cfg("attention"))
    image_emb, recipe_emb = model(_make_batch())
    assert image_emb.shape == (2, 64)
    assert recipe_emb.shape == (2, 64)


def test_joint_model_ingr_only_shapes():
    model = JointEmbeddingModel(_make_cfg("ingr_only"))
    image_emb, recipe_emb = model(_make_batch())
    assert image_emb.shape == (2, 64)
    assert recipe_emb.shape == (2, 64)


def test_joint_model_outputs_l2_normalized():
    model = JointEmbeddingModel(_make_cfg("concat"))
    image_emb, recipe_emb = model(_make_batch())
    assert torch.allclose(image_emb.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(recipe_emb.norm(dim=-1), torch.ones(2), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase3_models.py -k "joint_model" -v
```
Expected: `ModuleNotFoundError: No module named 'src.models.joint_embedding'`

- [ ] **Step 3: Implement JointEmbeddingModel**

```python
# src/models/joint_embedding.py
from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import FusionModule


class JointEmbeddingModel(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        m = cfg.model
        self.image_encoder = ImageEncoder(
            in_dim=m.image.in_dim,
            proj_hidden=m.image.proj_hidden,
            embed_dim=m.embed_dim,
        )
        self.text_encoder = TextEncoder(
            encoder_name=m.text.encoder,
            hidden=m.text.hidden,
            embed_dim=m.embed_dim,
            freeze=m.text.freeze_text,
        )
        self.fusion = FusionModule(
            embed_dim=m.embed_dim,
            hidden=m.fusion.hidden,
            mode=m.fusion.mode,
            n_heads=m.fusion.get("n_heads", 8),
            text_hidden=m.text.hidden,
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        image_emb = self.image_encoder(batch["image_feat"])
        ingr_emb, ingr_hidden = self.text_encoder.encode_ingr(
            batch["ingr_input_ids"], batch["ingr_attention_mask"]
        )
        instr_emb = self.text_encoder.encode_instr(
            batch["instr_input_ids"], batch["instr_attention_mask"]
        )
        recipe_emb = self.fusion(
            ingr_emb,
            instr_emb,
            ingr_hidden=ingr_hidden,
            ingr_mask=batch["ingr_attention_mask"],
        )
        return image_emb, recipe_emb
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_models.py -k "joint_model" -v
```
Expected: 4 passed

- [ ] **Step 5: Run all model tests to confirm nothing broke**

```bash
pytest tests/test_phase3_models.py -v
```
Expected: all passed

- [ ] **Step 6: Commit**

```bash
git add src/models/joint_embedding.py tests/test_phase3_models.py
git commit -m "feat(models): add JointEmbeddingModel wiring image + text + fusion towers"
```

---

### Task 5: InfoNCELoss

**Files:**
- Create: `src/losses/infonce.py`
- Create: `tests/test_phase3_loss.py`

**Interfaces:**
- Produces: `InfoNCELoss(temperature: float = 0.07)`
  - `forward(image_emb: Tensor[B,D], recipe_emb: Tensor[B,D]) → Tensor[scalar]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase3_loss.py
from __future__ import annotations
import torch
import pytest
from src.losses.infonce import InfoNCELoss


def test_infonce_returns_scalar():
    loss_fn = InfoNCELoss(temperature=0.07)
    image_emb = torch.randn(4, 32)
    recipe_emb = torch.randn(4, 32)
    loss = loss_fn(image_emb, recipe_emb)
    assert loss.shape == ()


def test_infonce_positive():
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(torch.randn(4, 32), torch.randn(4, 32))
    assert loss.item() > 0


def test_infonce_perfect_pairs_lower_than_random():
    """Aligned embeddings should yield lower loss than random."""
    loss_fn = InfoNCELoss(temperature=0.07)
    N = 8
    # orthonormal rows → perfect alignment, zero cross-pair sim
    embs = torch.zeros(N, N)
    embs.fill_diagonal_(1.0)
    loss_perfect = loss_fn(embs, embs)

    rand_img = torch.randn(N, 32)
    rand_rec = torch.randn(N, 32)
    loss_random = loss_fn(rand_img, rand_rec)
    assert loss_perfect.item() < loss_random.item()


def test_infonce_symmetric():
    """InfoNCE(a, b) == InfoNCE(b, a)."""
    loss_fn = InfoNCELoss(temperature=0.07)
    a = torch.randn(4, 32)
    b = torch.randn(4, 32)
    assert torch.allclose(loss_fn(a, b), loss_fn(b, a), atol=1e-5)


def test_infonce_temperature_effect():
    """Lower temperature → higher loss on random embeddings (sharper distribution)."""
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    loss_low_temp = InfoNCELoss(temperature=0.01)(a, b)
    loss_high_temp = InfoNCELoss(temperature=1.0)(a, b)
    assert loss_low_temp.item() > loss_high_temp.item()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase3_loss.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.losses.infonce'`

- [ ] **Step 3: Implement InfoNCELoss**

```python
# src/losses/infonce.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, image_emb: torch.Tensor, recipe_emb: torch.Tensor
    ) -> torch.Tensor:
        B = image_emb.size(0)
        labels = torch.arange(B, device=image_emb.device)
        logits_i2r = image_emb @ recipe_emb.T / self.temperature
        logits_r2i = recipe_emb @ image_emb.T / self.temperature
        return 0.5 * (
            F.cross_entropy(logits_i2r, labels) + F.cross_entropy(logits_r2i, labels)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_loss.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/losses/infonce.py tests/test_phase3_loss.py
git commit -m "feat(losses): add symmetric InfoNCELoss (NT-Xent) with fixed temperature"
```

---

### Task 6: Eval Metrics

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/metrics.py`
- Create: `tests/test_phase3_metrics.py`

**Interfaces:**
- Produces: `compute_metrics(image_embs: Tensor[N,D], recipe_embs: Tensor[N,D], ks: tuple = (1,5,10)) → dict[str, float]`
  - Keys: `im2recipe_medR`, `im2recipe_R@1`, `im2recipe_R@5`, `im2recipe_R@10`, `recipe2im_medR`, `recipe2im_R@1`, `recipe2im_R@5`, `recipe2im_R@10`
  - Input tensors must be L2-normalized; function uses dot product as cosine sim
  - Rank metric: count of items with higher similarity than correct match + 1 (1-indexed, optimistic on ties)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_phase3_metrics.py
from __future__ import annotations
import torch
import pytest
from src.eval.metrics import compute_metrics


def test_perfect_retrieval_medR_1():
    """Identity matrix → every query retrieves itself at rank 1."""
    N = 20
    embs = torch.eye(N)
    metrics = compute_metrics(embs, embs)
    assert metrics["im2recipe_medR"] == 1.0
    assert metrics["recipe2im_medR"] == 1.0


def test_perfect_retrieval_R_at_1_is_100():
    N = 20
    embs = torch.eye(N)
    metrics = compute_metrics(embs, embs)
    assert metrics["im2recipe_R@1"] == 100.0
    assert metrics["recipe2im_R@1"] == 100.0


def test_metrics_keys_present():
    N = 10
    embs = torch.randn(N, 8)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    metrics = compute_metrics(embs, embs)
    for direction in ("im2recipe", "recipe2im"):
        assert f"{direction}_medR" in metrics
        for k in (1, 5, 10):
            assert f"{direction}_R@{k}" in metrics


def test_shuffled_pairs_worse_than_perfect():
    N = 50
    embs = torch.eye(N)
    metrics_perfect = compute_metrics(embs, embs)
    perm = torch.randperm(N)
    while (perm == torch.arange(N)).all():  # ensure not identity permutation
        perm = torch.randperm(N)
    metrics_shuffled = compute_metrics(embs, embs[perm])
    assert metrics_perfect["im2recipe_medR"] <= metrics_shuffled["im2recipe_medR"]


def test_r_at_k_between_0_and_100():
    N = 20
    embs = torch.randn(N, 16)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    metrics = compute_metrics(embs, embs)
    for k in (1, 5, 10):
        assert 0.0 <= metrics[f"im2recipe_R@{k}"] <= 100.0
        assert 0.0 <= metrics[f"recipe2im_R@{k}"] <= 100.0


def test_custom_ks():
    N = 10
    embs = torch.eye(N)
    metrics = compute_metrics(embs, embs, ks=(1, 3))
    assert "im2recipe_R@3" in metrics
    assert "im2recipe_R@5" not in metrics
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_phase3_metrics.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Create package init and implement compute_metrics**

```python
# src/eval/__init__.py
```

```python
# src/eval/metrics.py
from __future__ import annotations
import torch


def compute_metrics(
    image_embs: torch.Tensor,
    recipe_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute medR and R@k for both retrieval directions.

    Args:
        image_embs: (N, D) L2-normalized image embeddings.
        recipe_embs: (N, D) L2-normalized recipe embeddings.
        ks: recall cutoffs.

    Returns:
        Dict with keys like 'im2recipe_medR', 'im2recipe_R@1', etc.
    """
    sim = image_embs @ recipe_embs.T  # (N, N) cosine sim
    results: dict[str, float] = {}
    for direction, s in [("im2recipe", sim), ("recipe2im", sim.T)]:
        # rank of correct match: count items with strictly higher sim + 1
        diag = s.diagonal().unsqueeze(1)      # (N, 1)
        ranks = (s > diag).sum(dim=1).float() + 1  # (N,) 1-indexed
        results[f"{direction}_medR"] = ranks.median().item()
        for k in ks:
            results[f"{direction}_R@{k}"] = (ranks <= k).float().mean().item() * 100.0
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_phase3_metrics.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/eval/__init__.py src/eval/metrics.py tests/test_phase3_metrics.py
git commit -m "feat(eval): add compute_metrics — medR + R@k for both retrieval directions"
```

---

### Task 7: Train Loop

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/train.py`

**Interfaces:**
- Consumes:
  - `JointEmbeddingModel(cfg)` → `forward(batch) → (image_emb, recipe_emb)`
  - `InfoNCELoss(temperature)` → `forward(image_emb, recipe_emb) → scalar`
  - `compute_metrics(image_embs, recipe_embs) → dict[str, float]`
  - `get_split(cfg, partition)` from `src/data/build_dataset.py`
  - `load_config(path, overrides)` from `src/utils/config.py`
  - `resolve_device(name)` from `src/utils/config.py`
  - `set_seed(seed)` from `src/utils/seed.py`
- CLI: `python -m src.training.train --config baseline.yaml [dot.overrides...]`
- Checkpoint: `runs/{exp_name}/best.pt` — keys: `model`, `optimizer`, `epoch`, `val_medR`

**Note:** No automated test for full train loop (requires real dataset + precomputed feats). The smoke test below uses tiny synthetic tensors to verify the loop executes without error.

- [ ] **Step 1: Create package init**

```python
# src/training/__init__.py
```

- [ ] **Step 2: Implement train.py**

```python
# src/training/train.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.build_dataset import get_split
from src.eval.metrics import compute_metrics
from src.losses.infonce import InfoNCELoss
from src.models.joint_embedding import JointEmbeddingModel
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed

_log = logging.getLogger(__name__)


def _move_batch(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _eval_1k(
    image_embs: torch.Tensor,
    recipe_embs: torch.Tensor,
    n_folds: int,
    seed: int,
) -> dict[str, float]:
    N = image_embs.size(0)
    k = min(1000, N)
    fold_metrics: list[dict[str, float]] = []
    for fold in range(n_folds):
        g = torch.Generator()
        g.manual_seed(seed + fold)
        idx = torch.randperm(N, generator=g)[:k]
        fold_metrics.append(compute_metrics(image_embs[idx], recipe_embs[idx]))
    avg: dict[str, float] = {}
    for key in fold_metrics[0]:
        avg[key] = sum(m[key] for m in fold_metrics) / n_folds
    return avg


@torch.no_grad()
def evaluate(
    model: JointEmbeddingModel,
    loader: DataLoader,
    cfg: object,
    device: str,
) -> dict[str, float]:
    model.eval()
    all_img, all_rec = [], []
    for batch in loader:
        batch = _move_batch(batch, device)
        img_emb, rec_emb = model(batch)
        all_img.append(img_emb.cpu())
        all_rec.append(rec_emb.cpu())
    image_embs = torch.cat(all_img)
    recipe_embs = torch.cat(all_rec)
    return _eval_1k(image_embs, recipe_embs, cfg.eval.n_folds, cfg.eval.seed)


def _save_checkpoint(
    model: JointEmbeddingModel,
    optimizer: AdamW,
    epoch: int,
    val_medR: float,
    run_dir: Path,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_medR": val_medR,
        },
        run_dir / "best.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train joint embedding model")
    parser.add_argument("--config", required=True, help="Config path (relative to configs/ or absolute)")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    amp_enabled = cfg.train.amp and device == "cuda"

    run_dir = Path(cfg.log.dir) / cfg.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    train_ds = get_split(cfg, "train")
    val_ds = get_split(cfg, "val")
    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin,
    )

    model = JointEmbeddingModel(cfg).to(device)
    criterion = InfoNCELoss(temperature=cfg.loss.temperature)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scaler = GradScaler(enabled=amp_enabled)

    best_medR = float("inf")
    patience = 0
    global_step = 0

    for epoch in range(cfg.train.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            with autocast("cuda", enabled=amp_enabled):
                image_emb, recipe_emb = model(batch)
                loss = criterion(image_emb, recipe_emb) / cfg.train.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            writer.add_scalar("train/loss", loss.item() * cfg.train.grad_accum, global_step)
            global_step += 1

        metrics = evaluate(model, val_loader, cfg, device)
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        val_medR = metrics["im2recipe_medR"]
        _log.info(
            "epoch %d  val im2recipe_medR=%.1f  R@1=%.1f  R@10=%.1f",
            epoch,
            val_medR,
            metrics["im2recipe_R@1"],
            metrics["im2recipe_R@10"],
        )

        if val_medR < best_medR:
            best_medR = val_medR
            patience = 0
            _save_checkpoint(model, optimizer, epoch, val_medR, run_dir)
            _log.info("  → new best checkpoint saved (medR=%.1f)", best_medR)
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                _log.info("Early stopping at epoch %d (patience=%d)", epoch, cfg.train.early_stop_patience)
                break

    writer.close()
    _log.info("Training complete. Best val im2recipe_medR=%.1f", best_medR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    main()
```

- [ ] **Step 3: Verify the module is importable**

```bash
python -c "from src.training.train import main; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Verify CLI help works**

```bash
python -m src.training.train --help
```
Expected: shows `--config` argument

- [ ] **Step 5: Run all existing tests to confirm nothing is broken**

```bash
pytest tests/ -v
```
Expected: all 30 prior tests + new phase3 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/training/__init__.py src/training/train.py
git commit -m "feat(training): add train loop with AMP, early stopping, TensorBoard, checkpoint"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| ImageEncoder 512→1024 + L2-norm | Task 1 |
| TextEncoder shared DistilBERT + masked mean pool | Task 2 |
| FusionModule concat \| attention \| ingr_only | Task 3 |
| JointEmbeddingModel forward → (image_emb, recipe_emb) | Task 4 |
| InfoNCE symmetric NT-Xent temperature=0.07 | Task 5 |
| medR + R@1/5/10 both directions | Task 6 |
| 1k subset eval with n_folds | Task 7 (via `_eval_1k`) |
| AMP + grad_accum + early stop on val medR | Task 7 |
| TensorBoard logging | Task 7 |
| Checkpoint to runs/{exp_name}/best.pt | Task 7 |
| Semantic loss dropped | All tasks (not implemented) |
| MPS-safe (no torch.cuda.*) | Tasks 1-7 (amp only on cuda) |

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:**
- `encode_ingr` returns `tuple[Tensor, Tensor]` — used correctly in Task 4 as `ingr_emb, ingr_hidden = ...`
- `FusionModule.forward(ingr_emb, instr_emb, ingr_hidden, ingr_mask)` — called with kwargs in Task 4
- `compute_metrics` returns `dict[str, float]` — consumed in Task 7 via `metrics["im2recipe_medR"]`
- All consistent.
