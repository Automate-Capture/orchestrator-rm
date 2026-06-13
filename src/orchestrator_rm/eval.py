from orchestrator_rm.data_utils import SyntheticDataset
from orchestrator_rm.reward_model import OrchestratorRewardModel


class Evaluator:
    """Pairwise ranking evaluator for a trained reward model."""

    def __init__(self, reward_model: OrchestratorRewardModel):
        self.reward_model = reward_model

    def evaluate_pairwise(self, trace_a: dict, trace_b: dict) -> dict:
        score_a = self.reward_model.score(trace_a)
        score_b = self.reward_model.score(trace_b)
        return {
            "score_a": score_a,
            "score_b": score_b,
            "winner": "a" if score_a > score_b else "b",
            "margin": abs(score_a - score_b),
        }

    def run_acceptance_test(self) -> bool:
        """Efficient trace must score higher than inefficient trace."""
        dataset = SyntheticDataset(seed=0)
        efficient, inefficient = dataset.make_contrastive_pair()
        result = self.evaluate_pairwise(efficient, inefficient)
        return result["score_a"] > result["score_b"]
