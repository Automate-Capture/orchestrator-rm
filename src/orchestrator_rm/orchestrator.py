import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from orchestrator_rm.reward_model import OrchestratorRewardModel
from orchestrator_rm.trace_collector import TraceCollector


class Orchestrator(nn.Module):
    """Policy network that selects the next sub-agent given a partial trace.

    Uses the reward model as a critic in a REINFORCE-style loop.
    """

    def __init__(
        self,
        agent_names: List[str],
        d_model: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.agent_names = list(agent_names)
        self.num_agents = len(self.agent_names)

        input_dim = 2 + self.num_agents
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_agents),
        )
        self._max_steps = 20
        self._max_tokens = 1000

    def _extract_features(self, partial_trace: dict) -> torch.Tensor:
        events = partial_trace.get("events", [])
        metadata = partial_trace.get("metadata", {})

        num_steps = len(events) / self._max_steps
        total_tokens = metadata.get("total_tokens", 0) / self._max_tokens

        last_agent = torch.zeros(self.num_agents)
        for e in reversed(events):
            name = e.get("agent", "")
            if name in self.agent_names:
                last_agent[self.agent_names.index(name)] = 1.0
                break

        return torch.cat([torch.tensor([num_steps, total_tokens]), last_agent])

    def forward(self, partial_trace: dict) -> torch.Tensor:
        features = self._extract_features(partial_trace)
        return self.policy(features)

    def select_agent(
        self, partial_trace: dict, temperature: float = 1.0
    ) -> Tuple[str, torch.Tensor]:
        logits = self.forward(partial_trace)
        probs = F.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return self.agent_names[action.item()], dist.log_prob(action)

    def run(
        self,
        query: str,
        collector: TraceCollector,
        max_steps: int = 3,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[dict, List[torch.Tensor]]:
        rng = random.Random(seed)
        trace: dict = {
            "query": query,
            "events": [],
            "outcome": "",
            "metadata": {"total_tokens": 0, "total_calls": 0},
        }

        log_probs: List[torch.Tensor] = []
        for _ in range(max_steps):
            agent_name, log_prob = self.select_agent(trace, temperature)
            log_probs.append(log_prob)
            trace = collector.step(trace, agent_name, rng=rng)

        return trace, log_probs

    def policy_gradient_step(
        self,
        traces_and_logprobs: List[Tuple[dict, List[torch.Tensor]]],
        reward_model: OrchestratorRewardModel,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        total_loss = torch.tensor(0.0)
        for trace, log_probs in traces_and_logprobs:
            reward = reward_model.score(trace)
            policy_loss = -reward * torch.stack(log_probs).sum()
            total_loss = total_loss + policy_loss

        avg_loss = total_loss / max(len(traces_and_logprobs), 1)
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        return avg_loss.item()
