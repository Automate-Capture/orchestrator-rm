from typing import List, Optional

import torch

from orchestrator_rm.cost_metric import CostMetric
from orchestrator_rm.data_utils import SyntheticDataset
from orchestrator_rm.orchestrator import Orchestrator
from orchestrator_rm.pair_generator import PairGenerator
from orchestrator_rm.reward_model import OrchestratorRewardModel


class OrchestrationTrainer:
    """Full pipeline: generate traces -> pair -> train reward model -> fine-tune policy."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.seed = seed

        self.reward_model: Optional[OrchestratorRewardModel] = None
        self.orchestrator: Optional[Orchestrator] = None
        self.dataset = SyntheticDataset(seed=seed)

    def train(
        self,
        num_queries: int = 50,
        rm_epochs: int = 10,
        policy_epochs: int = 5,
        rm_lr: float = 1e-3,
        policy_lr: float = 1e-4,
    ) -> dict:
        torch.manual_seed(self.seed)

        traces = self.dataset.make_dataset(num_queries=num_queries)

        cost_metric = CostMetric()
        pair_gen = PairGenerator(cost_metric=cost_metric)
        pairs = pair_gen.generate_pairs(traces)
        if not pairs:
            raise ValueError("No valid pairs generated from traces")

        self.reward_model = OrchestratorRewardModel(
            d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers
        )
        rm_losses = self.reward_model.fit(pairs, epochs=rm_epochs, lr=rm_lr)

        self.orchestrator = Orchestrator(
            self.dataset.agent_names,
            d_model=self.d_model,
            hidden_dim=self.hidden_dim,
        )
        policy_losses = self._train_policy(policy_epochs, policy_lr)

        return {
            "rm_losses": rm_losses,
            "policy_losses": policy_losses,
            "num_pairs": len(pairs),
            "num_traces": len(traces),
        }

    def _train_policy(self, epochs: int, lr: float) -> List[float]:
        collector = self.dataset.collector
        optimizer = torch.optim.Adam(self.orchestrator.parameters(), lr=lr)

        query_subset = self.dataset.queries[:10]
        policy_losses: List[float] = []

        for epoch in range(epochs):
            traces_and_logprobs = []
            for query in query_subset:
                trace, log_probs = self.orchestrator.run(
                    query, collector, max_steps=3, seed=epoch
                )
                traces_and_logprobs.append((trace, log_probs))

            loss = self.orchestrator.policy_gradient_step(
                traces_and_logprobs, self.reward_model, optimizer
            )
            policy_losses.append(loss)

        return policy_losses
