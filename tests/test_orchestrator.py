import torch

from orchestrator_rm.orchestrator import Orchestrator
from orchestrator_rm.reward_model import OrchestratorRewardModel
from orchestrator_rm.trace_collector import TraceCollector


def _echo_handler(query, ctx):
    return {"tokens": 10, "output_tokens": 5, "content": "ok"}


def _make_collector():
    tc = TraceCollector()
    tc.register_agent("a", _echo_handler)
    tc.register_agent("b", _echo_handler)
    tc.register_agent("c", _echo_handler)
    return tc


def test_select_agent_returns_valid():
    torch.manual_seed(0)
    orch = Orchestrator(["a", "b", "c"], d_model=16)
    trace = {"events": [], "metadata": {"total_tokens": 0, "total_calls": 0}}
    agent, log_prob = orch.select_agent(trace)
    assert agent in ["a", "b", "c"]
    assert log_prob.shape == ()


def test_run_produces_trace():
    torch.manual_seed(0)
    orch = Orchestrator(["a", "b", "c"], d_model=16)
    collector = _make_collector()
    trace, log_probs = orch.run("test", collector, max_steps=3, seed=0)

    assert len(log_probs) == 3
    assert trace["metadata"]["total_calls"] == 3
    assert len(trace["events"]) == 9  # 3 steps × 3 events each


def test_policy_gradient_step():
    torch.manual_seed(0)
    orch = Orchestrator(["a", "b", "c"], d_model=16)
    rm = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
    collector = _make_collector()
    optimizer = torch.optim.Adam(orch.parameters(), lr=1e-3)

    trace, log_probs = orch.run("test", collector, max_steps=2, seed=0)
    loss = orch.policy_gradient_step([(trace, log_probs)], rm, optimizer)
    assert isinstance(loss, float)


def test_deterministic_with_seed():
    torch.manual_seed(0)
    orch = Orchestrator(["a", "b", "c"], d_model=16)
    collector = _make_collector()

    t1, _ = orch.run("q", collector, max_steps=2, seed=99)
    t2, _ = orch.run("q", collector, max_steps=2, seed=99)
    assert t1["metadata"]["total_tokens"] == t2["metadata"]["total_tokens"]
