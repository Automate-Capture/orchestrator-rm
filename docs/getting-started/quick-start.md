# Quick Start

The following example runs end‑to‑end against the installed package:

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

For the full public API, see the [API Reference](../reference.md). For how the
pieces fit together, see [Architecture](../architecture.md).
