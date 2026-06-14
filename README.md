<p align="center">
  <img src="assets/hero.jpg" alt="OrchestrateRM" width="900">
</p>

<h1 align="center">OrchestrateRM</h1>

<p align="center"><strong>Self‑supervised Bradley–Terry reward model for multi‑agent orchestration.</strong></p>

<p align="center">
  <a href="https://github.com/Lumi-node/orchestrator-rm"><img src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github" alt="GitHub"></a>
  <a href="https://github.com/Lumi-node/orchestrator-rm/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://github.com/Lumi-node/orchestrator-rm/actions"><img src="https://img.shields.io/badge/tests-43-success.svg" alt="Tests"></a>
  <a href="https://lumi-node.github.io/orchestrator-rm/"><img src="https://img.shields.io/badge/docs-online-blue.svg" alt="Docs"></a>
</p>

---

OrchestrateRM learns a quality signal for multi‑agent orchestration from the agents' own execution traces — no human labels. It generates pairwise preferences from synthetic traces, fits a Bradley–Terry reward model over a lightweight transformer encoder, and exposes that reward for ranking traces or shaping orchestration policies.

## Installation

```bash
pip install git+https://github.com/Lumi-node/orchestrator-rm.git
```

Requires Python ≥ 3.10. To work on the project locally:

```bash
git clone https://github.com/Lumi-node/orchestrator-rm.git
cd orchestrator-rm
pip install -e ".[dev]"
pytest -q
```

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

## Features

- **Self‑supervised** — learns from automatically generated trace pairs, no human annotation
- **Bradley–Terry reward model** over a small transformer encoder
- **Synthetic trace & pair generation** utilities for reproducible experiments
- **Pairwise evaluator** for ranking efficient vs. inefficient orchestration

## Modules

| Module | Description |
|--------|-------------|
| `cost_metric` | — |
| `data_utils` | — |
| `encoder` | — |
| `eval` | — |
| `orchestrator` | — |
| `pair_generator` | — |
| `reward_model` | — |
| `trace_collector` | — |
| `trainer` | — |

## Documentation

📖 Full documentation: [https://lumi-node.github.io/orchestrator-rm/](https://lumi-node.github.io/orchestrator-rm/)
📄 Technical paper: see [`paper/`](paper/) for the LaTeX source and compiled PDF.

> This is a reference implementation produced by an autonomous research pipeline. It is not published to PyPI; install from source as shown above.

## License

[MIT](LICENSE) © Andrew Young / Automate Capture Research
