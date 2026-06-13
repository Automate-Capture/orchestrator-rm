from orchestrator_rm.cost_metric import CostMetric
from orchestrator_rm.pair_generator import PairGenerator


def _trace(outcome, tokens, calls):
    return {
        "outcome": outcome,
        "events": [],
        "metadata": {"total_tokens": tokens, "total_calls": calls},
    }


def test_pairs_same_outcome():
    pg = PairGenerator()
    t1 = _trace("42", 10, 1)
    t2 = _trace("42", 200, 5)
    pairs = pg.generate_pairs([t1, t2])

    assert len(pairs) == 1
    winner, loser = pairs[0]
    cm = CostMetric()
    assert cm.cost(winner) < cm.cost(loser)


def test_different_outcomes_no_cross_pairs():
    pg = PairGenerator()
    t1 = _trace("42", 10, 1)
    t2 = _trace("99", 200, 5)
    pairs = pg.generate_pairs([t1, t2])
    assert len(pairs) == 0


def test_multiple_traces_per_cluster():
    pg = PairGenerator()
    traces = [
        _trace("x", 10, 1),
        _trace("x", 50, 2),
        _trace("x", 200, 5),
    ]
    pairs = pg.generate_pairs(traces)
    assert len(pairs) == 3  # C(3,2) = 3


def test_single_trace_no_pairs():
    pg = PairGenerator()
    pairs = pg.generate_pairs([_trace("a", 10, 1)])
    assert len(pairs) == 0


def test_empty_input():
    pg = PairGenerator()
    assert pg.generate_pairs([]) == []


def test_custom_cost_metric():
    cm = CostMetric(token_weight=0.0, call_penalty=1.0)
    pg = PairGenerator(cost_metric=cm)
    t1 = _trace("r", 999, 1)
    t2 = _trace("r", 1, 2)
    pairs = pg.generate_pairs([t1, t2])
    winner, loser = pairs[0]
    assert winner is t1  # fewer calls wins despite more tokens
