import os
import tempfile

import torch

from orchestrator_rm.reward_model import OrchestratorRewardModel


def _trace(events_count, tokens_per_event=10):
    events = [
        {"type": "tool_call", "agent": "a", "tokens": tokens_per_event, "step": i}
        for i in range(events_count)
    ]
    return {
        "events": events,
        "outcome": "ok",
        "metadata": {
            "total_tokens": events_count * tokens_per_event,
            "total_calls": events_count,
        },
    }


def test_score_returns_float():
    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
    s = model.score(_trace(3))
    assert isinstance(s, float)


def test_fit_reduces_loss():
    torch.manual_seed(42)
    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)

    winner = _trace(1, tokens_per_event=5)
    loser = _trace(6, tokens_per_event=50)
    pairs = [(winner, loser)] * 20

    losses = model.fit(pairs, epochs=15, lr=1e-3)
    assert losses[-1] < losses[0]


def test_fit_learns_preference():
    torch.manual_seed(42)
    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)

    winner = _trace(1, tokens_per_event=5)
    loser = _trace(6, tokens_per_event=50)
    pairs = [(winner, loser)] * 30

    model.fit(pairs, epochs=18, lr=1e-3)

    assert model.score(winner) > model.score(loser)


def test_save_load_roundtrip():
    torch.manual_seed(0)
    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
    trace = _trace(3)
    score_before = model.score(trace)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        model2 = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
        model2.load(path)
        assert abs(model2.score(trace) - score_before) < 1e-6
    finally:
        os.unlink(path)
