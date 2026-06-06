"""Phase 0 verify: configs load, seed is reproducible, device resolves."""
import numpy as np

from src.utils.config import load_config, resolve_device
from src.utils.seed import set_seed


def test_baseline_config_loads_with_data_include():
    cfg = load_config("baseline.yaml")
    assert cfg.model.embed_dim == 1024
    assert cfg.data.subset.n_recipes == 2000          # from data.yaml include
    assert cfg.model.fusion.mode == "concat"


def test_cli_override_changes_resolved_config():
    cfg = load_config("baseline.yaml", overrides=["train.lr=5e-5", "data.subset.n_recipes=50000"])
    assert cfg.train.lr == 5e-5
    assert cfg.data.subset.n_recipes == 50000


def test_fusion_config_uses_attention():
    cfg = load_config("fusion.yaml")
    assert cfg.model.fusion.mode == "attention"
    assert cfg.model.fusion.pool == "attention"


def test_seed_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.allclose(a, b)


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
