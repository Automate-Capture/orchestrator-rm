from orchestrator_rm.cost_metric import CostMetric


def _make_trace(total_tokens, total_calls):
    return {"metadata": {"total_tokens": total_tokens, "total_calls": total_calls}}


def test_default_weights():
    cm = CostMetric()
    trace = _make_trace(50, 2)
    assert cm.cost(trace) == 50.0 * 1.0 + 2 * 100.0


def test_custom_weights():
    cm = CostMetric(token_weight=0.5, call_penalty=200.0)
    trace = _make_trace(100, 3)
    assert cm.cost(trace) == 100 * 0.5 + 3 * 200.0


def test_zero_trace():
    cm = CostMetric()
    trace = _make_trace(0, 0)
    assert cm.cost(trace) == 0.0


def test_missing_metadata():
    cm = CostMetric()
    assert cm.cost({}) == 0.0
    assert cm.cost({"metadata": {}}) == 0.0


def test_ordering():
    cm = CostMetric()
    cheap = _make_trace(10, 1)
    expensive = _make_trace(200, 5)
    assert cm.cost(cheap) < cm.cost(expensive)
