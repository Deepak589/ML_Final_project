from __future__ import annotations
import torch
import pytest
from src.losses.infonce import InfoNCELoss


def test_infonce_returns_scalar():
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(torch.randn(4, 32), torch.randn(4, 32))
    assert loss.shape == ()


def test_infonce_positive():
    loss_fn = InfoNCELoss(temperature=0.07)
    loss = loss_fn(torch.randn(4, 32), torch.randn(4, 32))
    assert loss.item() > 0


def test_infonce_perfect_pairs_lower_than_random():
    loss_fn = InfoNCELoss(temperature=0.07)
    N = 8
    embs = torch.eye(N, N)
    loss_perfect = loss_fn(embs, embs)
    rand_img = torch.randn(N, 32)
    rand_rec = torch.randn(N, 32)
    loss_random = loss_fn(rand_img, rand_rec)
    assert loss_perfect.item() < loss_random.item()


def test_infonce_symmetric():
    loss_fn = InfoNCELoss(temperature=0.07)
    a = torch.randn(4, 32)
    b = torch.randn(4, 32)
    assert torch.allclose(loss_fn(a, b), loss_fn(b, a), atol=1e-5)


def test_infonce_temperature_effect():
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    loss_low = InfoNCELoss(temperature=0.01)(a, b)
    loss_high = InfoNCELoss(temperature=1.0)(a, b)
    assert loss_low.item() > loss_high.item()
