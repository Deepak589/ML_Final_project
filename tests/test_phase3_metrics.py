from __future__ import annotations
import torch
import pytest
from src.eval.metrics import compute_metrics


def test_perfect_retrieval_medR_1():
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
    while (perm == torch.arange(N)).all():
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
