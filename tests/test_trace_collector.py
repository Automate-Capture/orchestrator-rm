import pytest

from orchestrator_rm.trace_collector import TraceCollector


def _echo_handler(query, ctx):
    return {"tokens": 10, "output_tokens": 5, "content": "ok", "tool": "echo"}


def _counter_handler(query, ctx):
    rng = ctx["rng"]
    return {"tokens": rng.randint(5, 15), "output_tokens": 3, "content": "counted"}


def test_register_and_collect():
    tc = TraceCollector()
    tc.register_agent("echo", _echo_handler)
    trace = tc.collect("hello", ["echo"], seed=42)

    assert trace["query"] == "hello"
    assert trace["outcome"] == "ok"
    assert len(trace["events"]) == 3  # select, tool_call, result
    assert trace["metadata"]["total_calls"] == 1
    assert trace["metadata"]["total_tokens"] == 10 + 5


def test_multi_agent_strategy():
    tc = TraceCollector()
    tc.register_agent("echo", _echo_handler)
    tc.register_agent("counter", _counter_handler)
    trace = tc.collect("test", ["echo", "counter"], seed=0)

    assert len(trace["events"]) == 6
    assert trace["metadata"]["total_calls"] == 2


def test_unknown_agent_raises():
    tc = TraceCollector()
    with pytest.raises(ValueError, match="Unknown agent"):
        tc.collect("test", ["nonexistent"])


def test_deterministic_with_seed():
    tc = TraceCollector()
    tc.register_agent("counter", _counter_handler)
    t1 = tc.collect("q", ["counter"], seed=123)
    t2 = tc.collect("q", ["counter"], seed=123)
    assert t1["metadata"]["total_tokens"] == t2["metadata"]["total_tokens"]


def test_step_method():
    tc = TraceCollector()
    tc.register_agent("echo", _echo_handler)

    trace = {
        "query": "hello",
        "events": [],
        "outcome": "",
        "metadata": {"total_tokens": 0, "total_calls": 0},
    }
    updated = tc.step(trace, "echo")

    assert updated["outcome"] == "ok"
    assert len(updated["events"]) == 3
    assert updated["metadata"]["total_calls"] == 1
    assert len(trace["events"]) == 0  # original unchanged


def test_agent_names_property():
    tc = TraceCollector()
    tc.register_agent("a", _echo_handler)
    tc.register_agent("b", _echo_handler)
    assert set(tc.agent_names) == {"a", "b"}
