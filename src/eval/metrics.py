from __future__ import annotations
import torch


def compute_metrics(
    image_embs: torch.Tensor,
    recipe_embs: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute medR and R@k for both retrieval directions.

    Args:
        image_embs: (N, D) L2-normalized image embeddings.
        recipe_embs: (N, D) L2-normalized recipe embeddings.
        ks: recall cutoffs.

    Returns:
        Dict with keys like 'im2recipe_medR', 'im2recipe_R@1', etc.
    """
    sim = image_embs @ recipe_embs.T  # (N, N) cosine sim
    results: dict[str, float] = {}
    for direction, s in [("im2recipe", sim), ("recipe2im", sim.T)]:
        diag = s.diagonal().unsqueeze(1)            # (N, 1)
        ranks = (s > diag).sum(dim=1).float() + 1  # (N,) 1-indexed
        results[f"{direction}_medR"] = ranks.median().item()
        for k in ks:
            results[f"{direction}_R@{k}"] = (ranks <= k).float().mean().item() * 100.0
    return results
