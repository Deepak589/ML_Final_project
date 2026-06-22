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
    autocast_device = "cuda" if device == "cuda" else "cpu"

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
            with autocast(autocast_device, enabled=amp_enabled):
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
            _log.info("  -> new best checkpoint saved (medR=%.1f)", best_medR)
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                _log.info("Early stopping at epoch %d", epoch)
                break

    writer.close()
    _log.info("Training complete. Best val im2recipe_medR=%.1f", best_medR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    main()
