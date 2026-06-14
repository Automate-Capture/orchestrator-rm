# OrchestrateRM

**Self‑supervised Bradley–Terry reward model for multi‑agent orchestration.**

OrchestrateRM learns a quality signal for multi‑agent orchestration from the agents' own execution traces — no human labels. It generates pairwise preferences from synthetic traces, fits a Bradley–Terry reward model over a lightweight transformer encoder, and exposes that reward for ranking traces or shaping orchestration policies.

## Quick Start

```python
import torch
from orchestrator_rm.cost_metric import CostMetric
from orchestrator_rm.data_utils import SyntheticDataset
from orchestrator_rm.pair_generator import PairGenerator
from orchestrator_rm.reward_model import OrchestratorRewardModel
from orchestrator_rm.eval import Evaluator

# 1. Build a synthetic dataset of orchestration traces
dataset = SyntheticDataset(seed=42)
traces = dataset.make_dataset(num_queries=16, traces_per_query=4)

# 2. Turn traces into preference pairs (cheaper trace preferred)
pairs = PairGenerator(CostMetric()).generate_pairs(traces)

# 3. Fit the Bradley–Terry reward model
model = OrchestratorRewardModel(d_model=16, nhead=4, num_layers=1)
model.fit(pairs, epochs=12, lr=1e-3)

# 4. Score: the model should rank the efficient trace above the inefficient one
efficient, inefficient = dataset.make_contrastive_pair()
result = Evaluator(model).evaluate_pairwise(efficient, inefficient)
print(result)   # {'winner': 'a', 'score_a': ..., 'score_b': ..., 'margin': ...}
```

See [Installation](getting-started/installation.md) and the [Quick Start guide](getting-started/quick-start.md) to go further, or the [API Reference](reference.md).
