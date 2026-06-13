from orchestrator_rm.data_utils import SyntheticDataset, STRATEGIES


def test_make_dataset_count():
    ds = SyntheticDataset(seed=0)
    traces = ds.make_dataset(num_queries=10, traces_per_query=4)
    assert len(traces) == 10 * 4


def test_traces_have_required_fields():
    ds = SyntheticDataset(seed=0)
    traces = ds.make_dataset(num_queries=5)
    for t in traces:
        assert "query" in t
        assert "events" in t
        assert "outcome" in t
        assert "metadata" in t
        assert "total_tokens" in t["metadata"]
        assert "total_calls" in t["metadata"]
        assert "strategy" in t


def test_contrastive_pair_same_outcome():
    ds = SyntheticDataset(seed=0)
    efficient, inefficient = ds.make_contrastive_pair()
    assert efficient["outcome"] == inefficient["outcome"]


def test_contrastive_pair_cost_difference():
    ds = SyntheticDataset(seed=0)
    efficient, inefficient = ds.make_contrastive_pair()
    assert efficient["metadata"]["total_tokens"] < inefficient["metadata"]["total_tokens"]
    assert efficient["metadata"]["total_calls"] < inefficient["metadata"]["total_calls"]


def test_collector_exposed_after_make_dataset():
    ds = SyntheticDataset(seed=0)
    ds.make_dataset(num_queries=5)
    collector = ds.collector
    assert set(collector.agent_names) == set(ds.agent_names)


def test_queries_stored():
    ds = SyntheticDataset(seed=0)
    ds.make_dataset(num_queries=7)
    assert len(ds.queries) == 7


def test_deterministic():
    ds1 = SyntheticDataset(seed=42)
    ds2 = SyntheticDataset(seed=42)
    t1 = ds1.make_dataset(num_queries=5)
    t2 = ds2.make_dataset(num_queries=5)
    for a, b in zip(t1, t2):
        assert a["query"] == b["query"]
        assert a["metadata"] == b["metadata"]


def test_all_strategies_represented():
    ds = SyntheticDataset(seed=0)
    traces = ds.make_dataset(num_queries=3, traces_per_query=4)
    strategies_seen = {t["strategy"] for t in traces}
    assert strategies_seen == set(STRATEGIES.keys())
