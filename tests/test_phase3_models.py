from __future__ import annotations
import torch
import pytest
from omegaconf import OmegaConf
from src.models.image_encoder import ImageEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import FusionModule
from src.models.joint_embedding import JointEmbeddingModel

# ── ImageEncoder ──────────────────────────────────────────────────────────────

def test_image_encoder_output_shape():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    out = enc(torch.randn(4, 512))
    assert out.shape == (4, 64)


def test_image_encoder_l2_normalized():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    out = enc(torch.randn(4, 512))
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_image_encoder_different_batch_sizes():
    enc = ImageEncoder(in_dim=512, proj_hidden=128, embed_dim=64)
    for B in (1, 8, 32):
        assert enc(torch.randn(B, 512)).shape == (B, 64)


# ── TextEncoder ───────────────────────────────────────────────────────────────

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


def test_text_encoder_freeze_flag():
    enc = TextEncoder(_ENC, hidden=_H, embed_dim=_D, freeze=True)
    for p in enc.bert.parameters():
        assert not p.requires_grad
    for p in enc.ingr_proj.parameters():
        assert p.requires_grad


# ── FusionModule ──────────────────────────────────────────────────────────────

_ED, _FH, _TH, _BF, _LF = 64, 128, 768, 4, 16


def test_fusion_concat_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="concat", text_hidden=_TH)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED))
    assert out.shape == (_BF, _ED)


def test_fusion_concat_l2_normalized():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="concat", text_hidden=_TH)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED))
    assert torch.allclose(out.norm(dim=-1), torch.ones(_BF), atol=1e-5)


def test_fusion_attention_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="attention", n_heads=4, text_hidden=_TH)
    out = fuse(
        torch.randn(_BF, _ED),
        torch.randn(_BF, _ED),
        ingr_hidden=torch.randn(_BF, _LF, _TH),
        ingr_mask=torch.ones(_BF, _LF, dtype=torch.long),
    )
    assert out.shape == (_BF, _ED)


def test_fusion_attention_l2_normalized():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="attention", n_heads=4, text_hidden=_TH)
    out = fuse(
        torch.randn(_BF, _ED),
        torch.randn(_BF, _ED),
        ingr_hidden=torch.randn(_BF, _LF, _TH),
        ingr_mask=torch.ones(_BF, _LF, dtype=torch.long),
    )
    assert torch.allclose(out.norm(dim=-1), torch.ones(_BF), atol=1e-5)


def test_fusion_ingr_only_shape():
    fuse = FusionModule(embed_dim=_ED, hidden=_FH, mode="ingr_only", text_hidden=_TH)
    out = fuse(torch.randn(_BF, _ED), torch.randn(_BF, _ED))
    assert out.shape == (_BF, _ED)


def test_fusion_invalid_mode_raises():
    with pytest.raises(ValueError, match="Unknown fusion mode"):
        FusionModule(embed_dim=_ED, hidden=_FH, mode="bad_mode", text_hidden=_TH)


# ── JointEmbeddingModel ───────────────────────────────────────────────────────

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
