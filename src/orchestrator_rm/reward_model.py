from typing import List, Tuple

import torch
import torch.nn as nn

from orchestrator_rm.encoder import TraceEncoder


class OrchestratorRewardModel(nn.Module):
    """Bradley-Terry reward model: learns to score traces so efficient ones rank higher."""

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.encoder = TraceEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, trace: dict) -> torch.Tensor:
        encoded = self.encoder.encode(trace)
        return self.head(encoded).squeeze(-1)

    def score(self, trace: dict) -> float:
        self.eval()
        with torch.no_grad():
            return self.forward(trace).item()

    def fit(
        self,
        pairs: List[Tuple[dict, dict]],
        epochs: int = 10,
        lr: float = 1e-3,
    ) -> List[float]:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses: List[float] = []

        for _epoch in range(epochs):
            epoch_loss = 0.0
            for winner, loser in pairs:
                s_w = self.forward(winner)
                s_l = self.forward(loser)
                loss = -torch.log(torch.sigmoid(s_w - s_l) + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            losses.append(epoch_loss / max(len(pairs), 1))

        return losses

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "agent_vocab": self.encoder._agent_vocab,
                "agent_counter": self.encoder._agent_counter,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.encoder._agent_vocab = checkpoint["agent_vocab"]
        self.encoder._agent_counter = checkpoint["agent_counter"]
