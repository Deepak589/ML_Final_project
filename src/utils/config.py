"""Config loading via OmegaConf. CLI flags override config keys only."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(path: str, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config, resolve a `defaults: {data: data.yaml}` include, apply CLI overrides.

    Args:
        path: config path (absolute, or relative to configs/).
        overrides: dotlist overrides, e.g. ["train.lr=5e-5", "subset.n_recipes=50000"].

    Returns:
        Merged DictConfig.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = _CONFIG_DIR / path
    cfg = OmegaConf.load(cfg_path)

    # resolve a single-level `defaults` include (data.yaml) into a `data:` node
    defaults = cfg.pop("defaults", None)
    if defaults is not None and "data" in defaults:
        data_cfg = OmegaConf.load(_CONFIG_DIR / defaults["data"])
        cfg = OmegaConf.merge({"data": data_cfg}, cfg)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def resolve_device(name: str) -> str:
    """Map 'auto' -> cuda|mps|cpu; pass through explicit names."""
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
