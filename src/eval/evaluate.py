"""Evaluate a saved checkpoint on the test (or val) split."""
from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader

from src.data.build_dataset import get_split
from src.eval.metrics import compute_metrics
from src.models.joint_embedding import JointEmbeddingModel
from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed

_log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model: JointEmbeddingModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    all_img, all_rec = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        img_emb, rec_emb = model(batch)
        all_img.append(img_emb.cpu())
        all_rec.append(rec_emb.cpu())
    return compute_metrics(torch.cat(all_img), torch.cat(all_rec))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    ds = get_split(cfg, args.split)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    model = JointEmbeddingModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    _log.info("Loaded checkpoint from epoch %d (val medR=%.1f)", ckpt["epoch"], ckpt["val_medR"])

    metrics = evaluate(model, loader, device)

    print(f"\n=== {args.split.upper()} SET RESULTS ===")
    for direction in ("im2recipe", "recipe2im"):
        print(f"\n{direction}:")
        print(f"  medR   : {metrics[f'{direction}_medR']:.1f}")
        for k in (1, 5, 10):
            print(f"  R@{k:<3}  : {metrics[f'{direction}_R@{k}']:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    main()
