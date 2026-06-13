import torch

from orchestrator_rm.cost_metric import CostMetric
from orchestrator_rm.data_utils import SyntheticDataset
from orchestrator_rm.eval import Evaluator
from orchestrator_rm.pair_generator import PairGenerator
from orchestrator_rm.reward_model import OrchestratorRewardModel


def test_acceptance_pairwise_ranking():
    """Core acceptance test: after training, the model ranks efficient > inefficient."""
    torch.manual_seed(42)

    dataset = SyntheticDataset(seed=42)
    traces = dataset.make_dataset(num_queries=16, traces_per_query=4)

    pair_gen = PairGenerator(CostMetric())
    pairs = pair_gen.generate_pairs(traces)
    assert len(pairs) > 50

    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
    losses = model.fit(pairs, epochs=30, lr=1e-3)
    assert losses[-1] < losses[0]

    efficient, inefficient = dataset.make_contrastive_pair()

    evaluator = Evaluator(model)
    result = evaluator.evaluate_pairwise(efficient, inefficient)

    assert result["winner"] == "a"
    assert result["margin"] > 0


def test_evaluator_run_acceptance():
    """Convenience wrapper also passes."""
    torch.manual_seed(42)

    dataset = SyntheticDataset(seed=42)
    traces = dataset.make_dataset(num_queries=16, traces_per_query=4)

    model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
    pairs = PairGenerator().generate_pairs(traces)
    model.fit(pairs, epochs=12, lr=1e-3)

    evaluator = Evaluator(model)
    assert evaluator.run_acceptance_test()
