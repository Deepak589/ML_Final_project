"""Precompute CLIP ViT-B-32 image features and save {stem: Tensor(512)} dict."""
from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config, resolve_device


def main(cfg: DictConfig) -> None:
    images_dir = Path(cfg.data.paths.images)
    out_path = Path(cfg.data.paths.image_feats)
    batch_size: int = cfg.get("batch_size", 64)
    device = resolve_device(cfg.get("device", "cpu"))

    jpg_paths = sorted(images_dir.glob("*.jpg"))
    if not jpg_paths:
        raise FileNotFoundError(f"No .jpg files found in {images_dir}")
    print(f"Found {len(jpg_paths)} images in {images_dir}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg.data.image.clip_model,
        pretrained=cfg.data.image.clip_pretrained,
        device=device,
    )
    model.eval()

    feats_dict: dict[str, torch.Tensor] = {}
    skipped = 0

    for i in tqdm(range(0, len(jpg_paths), batch_size), desc="Encoding"):
        batch_paths = jpg_paths[i : i + batch_size]
        imgs, stems = [], []
        for p in batch_paths:
            try:
                imgs.append(preprocess(Image.open(p).convert("RGB")))
                stems.append(p.stem)
            except Exception as exc:
                print(f"Warning: skipping {p.name}: {exc}")
                skipped += 1
        if not imgs:
            continue
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for stem, feat in zip(stems, feats.cpu()):
            feats_dict[stem] = feat

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats_dict, out_path)
    print(f"Saved {len(feats_dict)} features to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute CLIP image features")
    parser.add_argument("--config", default="baseline.yaml")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.batch_size = args.batch_size
    if args.device:
        cfg.device = args.device
    main(cfg)
