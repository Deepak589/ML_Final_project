"""Query top-K recipes from a single food image.

Usage:
    python -m src.eval.demo \
        --config baseline.yaml \
        --checkpoint runs/baseline_concat/best.pt \
        --image path/to/food.jpg \
        --topk 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.build_dataset import get_split
from src.models.joint_embedding import JointEmbeddingModel
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed

_log = logging.getLogger(__name__)


def embed_query_image(image_path: str, device: str) -> torch.Tensor:
    """Run CLIP ViT-B/32 on a raw image → 512-d feat."""
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
    return F.normalize(feat.float(), dim=-1).cpu()


@torch.no_grad()
def build_recipe_index(
    model: JointEmbeddingModel, loader: DataLoader, device: str, ds
) -> tuple[torch.Tensor, list[dict]]:
    model.eval()
    all_embs = []
    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        _, rec_emb = model(batch_dev)
        all_embs.append(rec_emb.cpu())
    metadata = [
        {
            "title": r.get("title", r.get("id", "?")),
            "ingredients": ", ".join(r["ingredients"]),
            "instructions": r.get("instructions", ""),
        }
        for r in ds.recipes
    ]
    return torch.cat(all_embs), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Path to query food image")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--topk", type=int, default=5)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    # Load model
    model = JointEmbeddingModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    _log.info("Loaded checkpoint (epoch %d, val medR=%.1f)", ckpt["epoch"], ckpt["val_medR"])

    # Embed query image
    print(f"\nEmbedding query image: {args.image}")
    clip_feat = embed_query_image(args.image, device).to(device)
    with torch.no_grad():
        query_emb = model.image_encoder(clip_feat).cpu()  # (1, D)

    # Build recipe index from split
    print(f"Building recipe index from {args.split} split...")
    ds = get_split(cfg, args.split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    recipe_embs, metadata = build_recipe_index(model, loader, device, ds)

    # Ground truth lookup by image stem
    query_stem = Path(args.image).stem
    ground_truth = next(
        (r for r in ds.recipes if Path(r["image_path"]).stem == query_stem), None
    )
    if ground_truth:
        print(f"\n=== GROUND TRUTH ===")
        print(f"  Title      : {ground_truth['title']}")
        print(f"  Ingredients: {', '.join(ground_truth['ingredients'])}")
    else:
        print(f"\n[!] No ground truth found for stem '{query_stem}' in {args.split} split")

    # Cosine search
    sims = (query_emb @ recipe_embs.T).squeeze(0)  # (N,)
    topk_idx = sims.topk(args.topk).indices.tolist()

    print(f"\n=== TOP {args.topk} MODEL PREDICTIONS ===")
    for rank, idx in enumerate(topk_idx, 1):
        m = metadata[idx]
        match = " ✓ CORRECT" if ground_truth and m["title"] == ground_truth["title"] else ""
        print(f"\n{'='*60}")
        print(f"#{rank}  score={sims[idx]:.3f}  |  {m['title']}{match}")
        print(f"\nIngredients:\n  {m['ingredients']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    main()
