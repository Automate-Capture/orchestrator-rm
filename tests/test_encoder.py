import torch

from orchestrator_rm.encoder import TraceEncoder, EVENT_TYPES


def _make_trace(n_events=3):
    events = []
    types = ["select", "tool_call", "result"]
    for i in range(n_events):
        events.append({
            "type": types[i % len(types)],
            "agent": "agent_a",
            "tokens": (i + 1) * 10,
            "step": i,
        })
    return {
        "query": "test",
        "events": events,
        "outcome": "done",
        "metadata": {"total_tokens": sum(e["tokens"] for e in events), "total_calls": 1},
    }


def test_encode_shape():
    enc = TraceEncoder(d_model=32, nhead=4, num_layers=1)
    trace = _make_trace(6)
    vec = enc.encode(trace)
    assert vec.shape == (32,)


def test_encode_batch_shape():
    enc = TraceEncoder(d_model=32, nhead=4, num_layers=1)
    traces = [_make_trace(3), _make_trace(6), _make_trace(9)]
    batch = enc.encode_batch(traces)
    assert batch.shape == (3, 32)


def test_empty_trace():
    enc = TraceEncoder(d_model=16, nhead=4, num_layers=1)
    trace = {"events": [], "metadata": {}}
    vec = enc.encode(trace)
    assert vec.shape == (16,)


def test_agent_vocab_growth():
    enc = TraceEncoder(d_model=16, nhead=4, num_layers=1, max_agents=4)
    assert enc.get_agent_id("alpha") == 1
    assert enc.get_agent_id("beta") == 2
    assert enc.get_agent_id("alpha") == 1  # stable
    assert enc.get_agent_id("gamma") == 3
    assert enc.get_agent_id("overflow") == 0  # max_agents=4, slots 0-3 used


def test_different_traces_different_vectors():
    torch.manual_seed(0)
    enc = TraceEncoder(d_model=32, nhead=4, num_layers=1)
    v1 = enc.encode(_make_trace(3))
    v2 = enc.encode(_make_trace(9))
    assert not torch.allclose(v1, v2)


def test_tokenize_all_event_types():
    enc = TraceEncoder(d_model=16, nhead=4, num_layers=1)
    events = [{"type": t, "agent": "x", "tokens": 1, "step": i}
              for i, t in enumerate(EVENT_TYPES) if t != "pad"]
    trace = {"events": events, "metadata": {}}
    tokens = enc.tokenize_trace(trace)
    assert len(tokens["event_type_ids"]) == len(events)
