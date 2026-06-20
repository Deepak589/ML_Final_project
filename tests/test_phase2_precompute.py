"""Phase 2: precompute_image_feats.main() contract tests. CLIP is mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


@pytest.fixture
def tmp_images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("alpha", "beta", "gamma"):
        Image.new("RGB", (1, 1), color=(128, 64, 32)).save(img_dir / f"{name}.jpg")
    return img_dir


@pytest.fixture
def tmp_cfg(tmp_path, tmp_images):
    return OmegaConf.create({
        "data": {
            "paths": {
                "images": str(tmp_images),
                "image_feats": str(tmp_path / "processed" / "image_feats.pt"),
            },
            "image": {"clip_model": "ViT-B-32", "clip_pretrained": "openai"},
        },
        "device": "cpu",
        "batch_size": 64,
    })


def _make_mock_clip():
    model = MagicMock()
    model.encode_image.side_effect = lambda batch: torch.randn(batch.shape[0], 512)
    preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    return model, preprocess


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_output_file_created(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    assert Path(tmp_cfg.data.paths.image_feats).exists()


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_dict_keys_are_stems(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    assert set(feats.keys()) == {"alpha", "beta", "gamma"}


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_feat_shape_is_512(mock_create, tmp_cfg):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    from src.data.precompute_image_feats import main
    main(tmp_cfg)

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    for stem, feat in feats.items():
        assert isinstance(feat, torch.Tensor), f"{stem}: expected Tensor"
        assert feat.shape == (512,), f"{stem}: expected (512,), got {feat.shape}"


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_corrupt_image_skipped(mock_create, tmp_cfg, tmp_images):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    (tmp_images / "corrupt.jpg").write_bytes(b"not_a_jpeg")

    from src.data.precompute_image_feats import main
    main(tmp_cfg)  # must not raise

    feats = torch.load(tmp_cfg.data.paths.image_feats, weights_only=False)
    assert "corrupt" not in feats
    assert len(feats) == 3  # only the 3 valid images


@patch("src.data.precompute_image_feats.open_clip.create_model_and_transforms")
def test_no_images_raises(mock_create, tmp_path):
    model, preprocess = _make_mock_clip()
    mock_create.return_value = (model, None, preprocess)

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cfg = OmegaConf.create({
        "data": {
            "paths": {
                "images": str(empty_dir),
                "image_feats": str(tmp_path / "out.pt"),
            },
            "image": {"clip_model": "ViT-B-32", "clip_pretrained": "openai"},
        },
        "device": "cpu",
        "batch_size": 64,
    })

    from src.data.precompute_image_feats import main
    with pytest.raises(FileNotFoundError, match="No .jpg"):
        main(cfg)
