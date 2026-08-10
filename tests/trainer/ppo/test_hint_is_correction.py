import math

import pytest
import torch

from verl.trainer.ppo.ray_trainer import compute_hint_is_weights


def test_hint_is_weights_use_sequence_log_ratio_and_response_mask():
    old_log_probs = torch.tensor([[-1.0, -2.0, 99.0], [-3.0, -4.0, -5.0]])
    behavior_log_probs = torch.tensor([[-1.2, -2.3, -99.0], [-3.4, -4.5, -5.6]])
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    hint_mask = torch.tensor([True, False])

    weights, metrics = compute_hint_is_weights(
        old_log_probs,
        behavior_log_probs,
        response_mask,
        hint_mask,
        log_c_clip=5.0,
    )

    assert weights.shape == (2, 1)
    assert weights[0, 0].item() == pytest.approx(math.exp(0.5))
    assert weights[1, 0].item() == 1.0
    assert metrics["hint_is/log_c_mean"] == pytest.approx(0.5)
    assert metrics["hint_is/clip_frac"] == 0.0
    assert not weights.requires_grad


def test_hint_is_weights_clip_trajectory_log_ratio():
    old_log_probs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    behavior_log_probs = torch.tensor([[-4.0, -4.0], [4.0, 4.0]])
    response_mask = torch.ones_like(old_log_probs, dtype=torch.bool)

    weights, metrics = compute_hint_is_weights(
        old_log_probs,
        behavior_log_probs,
        response_mask,
        torch.tensor([True, True]),
        log_c_clip=2.0,
    )

    assert weights[:, 0].tolist() == pytest.approx([math.exp(2.0), math.exp(-2.0)])
    assert metrics["hint_is/log_c_min"] == pytest.approx(-8.0)
    assert metrics["hint_is/log_c_max"] == pytest.approx(8.0)
    assert metrics["hint_is/clip_frac"] == 1.0


def test_hint_is_weights_reject_nonpositive_clip():
    values = torch.zeros(1, 1)
    with pytest.raises(ValueError, match="hint_log_c_clip must be positive"):
        compute_hint_is_weights(values, values, torch.ones_like(values), torch.tensor([True]), 0.0)
