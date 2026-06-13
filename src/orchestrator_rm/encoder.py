import math
from typing import Dict, List

import torch
import torch.nn as nn

EVENT_TYPES = {"pad": 0, "select": 1, "tool_call": 2, "result": 3, "error": 4}
MAX_SEQ_LEN = 128


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        cols_odd = pe[:, 1::2].size(1)
        pe[:, 1::2] = torch.cos(position * div_term[:cols_odd])
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0)]


class TraceEncoder(nn.Module):
    """Maps a trace (sequence of events) to a fixed-size vector via Transformer."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_event_types: int = len(EVENT_TYPES),
        max_agents: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.event_type_embed = nn.Embedding(num_event_types, d_model)
        self.agent_embed = nn.Embedding(max_agents, d_model)
        self.numeric_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._agent_vocab: Dict[str, int] = {"<unk>": 0}
        self._agent_counter: int = 1

    def get_agent_id(self, agent_name: str) -> int:
        if agent_name not in self._agent_vocab:
            if self._agent_counter < self.agent_embed.num_embeddings:
                self._agent_vocab[agent_name] = self._agent_counter
                self._agent_counter += 1
            else:
                return 0
        return self._agent_vocab[agent_name]

    def tokenize_trace(self, trace: dict) -> dict:
        events = trace.get("events", [])
        if not events:
            events = [{"type": "pad", "agent": "<unk>", "tokens": 0, "step": 0}]

        event_type_ids = []
        agent_ids = []
        numerics = []

        max_tokens = max((e.get("tokens", 0) for e in events), default=1) or 1
        max_step = max((e.get("step", 0) for e in events), default=1) or 1

        for e in events:
            event_type_ids.append(EVENT_TYPES.get(e.get("type", "pad"), 0))
            agent_ids.append(self.get_agent_id(e.get("agent", "<unk>")))
            numerics.append([
                e.get("tokens", 0) / max_tokens,
                e.get("step", 0) / max_step,
            ])

        return {
            "event_type_ids": torch.tensor(event_type_ids, dtype=torch.long),
            "agent_ids": torch.tensor(agent_ids, dtype=torch.long),
            "numerics": torch.tensor(numerics, dtype=torch.float32),
        }

    def _encode_tokens(self, tokens: dict) -> torch.Tensor:
        evt_emb = self.event_type_embed(tokens["event_type_ids"])
        agt_emb = self.agent_embed(tokens["agent_ids"])
        num_emb = self.numeric_proj(tokens["numerics"])

        combined = (evt_emb + agt_emb + num_emb).unsqueeze(1)  # (seq, 1, d_model)
        combined = self.pos_enc(combined)
        encoded = self.transformer(combined)                     # (seq, 1, d_model)
        return encoded.mean(dim=0).squeeze(0)                    # (d_model,)

    def encode(self, trace: dict) -> torch.Tensor:
        tokens = self.tokenize_trace(trace)
        return self._encode_tokens(tokens)

    def encode_batch(self, traces: List[dict]) -> torch.Tensor:
        return torch.stack([self.encode(t) for t in traces])
