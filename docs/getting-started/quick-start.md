# Quick‑Start Guide – `orchestrator_rm`

Welcome! This guide shows you how to get the **Orchestrator Reward Model (RM)** up and running in just a few minutes.  
All the public API you’ll need is listed in the *Modules* section below – you only ever import from `orchestrator_rm.<module>`.

---

## 1️⃣ Install the package

```bash
# From PyPI (once the package is released)
pip install orchestrator_rm

# Or, install from source (recommended for development)
git clone https://github.com/your‑org/orchestrator_rm.git
cd orchestrator_rm
pip install -e .
```

The installation pulls in the required dependencies (`torch`, `numpy`, `tqdm`, …) automatically.

---

## 2️⃣ Overview of the core modules

| Module | What it provides |
|--------|-------------------|
| `cost_metric.py` | Simple cost function for a trace (`cost(trace) → float`). |
| `data_utils.py` | Helpers to generate synthetic queries, collect traces, and build contrastive datasets. |
| `encoder.py` | `TraceEncoder` (turns a trace dict into a vector) and `PositionalEncoding`. |
| `eval.py` | Pairwise evaluation of two traces (`evaluate_pairwise`) and a quick acceptance test (`run_acceptance_test`). |
| `orchestrator.py` | `Orchestrator` model that decides which agent to call next. |
| `pair_generator.py` | Turn a list of traces into contrastive pairs (`generate_pairs`). |
| `reward_model.py` | `OrchestratorRewardModel` – the Bradley‑Terry self‑supervised reward model. |
| `trace_collector.py` | `TraceCollector` – runs agents on a query and records the full execution trace. |
| `trainer.py` | High‑level training loop (`train`). |

Only the functions/classes listed in the **API** section are part of the public surface; everything else is internal.

---

## 3️⃣ Minimal end‑to‑end example

Below is a **complete, runnable script** that:

1. Creates a synthetic query set.  
2. Collects a few traces per query.  
3. Builds a contrastive dataset.  
4. Trains the reward model.  
5. Evaluates a random pair.

```python
# example_train_and_eval.py
import torch
from orchestrator_rm.data_utils import make_dataset, make_contrastive_pair
from orchestrator_rm.trace_collector import TraceCollector
from orchestrator_rm.orchestrator import Orchestrator
from orchestrator_rm.encoder import TraceEncoder
from orchestrator_rm.reward_model import OrchestratorRewardModel
from orchestrator_rm.trainer import train
from orchestrator_rm.eval import evaluate_pairwise

# ----------------------------------------------------------------------
# 1️⃣  Set up a trace collector (agents are dummy functions for the demo)
# ----------------------------------------------------------------------
collector = TraceCollector()

def dummy_agent_handler(query, ctx):
    """A trivial agent that echoes the query and adds a random token."""
    import random, string
    token = random.choice(string.ascii_lowercase)
    return {"output": f"{query}_{token}", "ctx": ctx}

# Register a few agents
for name in ["agent_a", "agent_b", "agent_c"]:
    collector.register_agent(name, dummy_agent_handler)

# ----------------------------------------------------------------------
# 2️⃣  Generate a small synthetic dataset
# ----------------------------------------------------------------------
dataset = make_dataset(num_queries=20, traces_per_query=3)   # → List[dict]

# ----------------------------------------------------------------------
# 3️⃣  Build a contrastive pair (trace_a is “better” than trace_b)
# ----------------------------------------------------------------------
trace_a, trace_b = make_contrastive_pair(dataset)

# ----------------------------------------------------------------------
# 4️⃣  Initialise the models
# ----------------------------------------------------------------------
encoder = TraceEncoder()
reward_model = OrchestratorRewardModel(encoder)

# ----------------------------------------------------------------------
# 5️⃣  Train the reward model (single‑step demo)
# ----------------------------------------------------------------------
train(
    reward_model=reward_model,
    trace_pairs=[(trace_a, trace_b)],   # list of (better, worse) pairs
    optimizer=torch.optim.Adam(reward_model.parameters(), lr=1e-3),
    epochs=5,
)

# ----------------------------------------------------------------------
# 6️⃣  Evaluate the learned model on the same pair
# ----------------------------------------------------------------------
result = evaluate_pairwise(trace_a, trace_b)
print("Pairwise evaluation:", result)
# Example output:
# {'p_a_wins': 0.73, 'p_b_wins': 0.27, 'logit': 1.03}
```

### What the script does

| Step | API call | Purpose |
|------|----------|---------|
| 1 | `TraceCollector.register_agent` | Register dummy agents that the orchestrator can call. |
| 2 | `make_dataset` | Synthesize a list of trace dictionaries (each trace contains the full sequence of agent calls). |
| 3 | `make_contrastive_pair` | Pull two traces from the dataset and label the first as the “preferred” one. |
| 4 | `TraceEncoder` + `OrchestratorRewardModel` | Build the neural reward model (encoder → Bradley‑Terry head). |
| 5 | `train` | Run a tiny training loop that updates the reward model using the pairwise Bradley‑Terry loss. |
| 6 | `evaluate_pairwise` | Convert the learned logits into win probabilities for the two traces. |

---

## 4️⃣ Quick‑start with the **Orchestrator** itself

If you want to see the orchestrator in action (i.e., how it selects the next agent given a partial trace), use the following snippet:

```python
# example_orchestrator.py
import torch
from orchestrator_rm.orchestrator import Orchestrator
from orchestrator_rm.trace_collector import TraceCollector

# Re‑use the collector from the previous example
collector = TraceCollector()
# … (register agents as before) …

# Initialise the orchestrator (it internally uses a TraceEncoder)
orch = Orchestrator()

# Start a new query
query = "order pizza"
partial = {"query": query, "steps": []}

# Let the orchestrator pick an agent and run one step
selected = orch.select_agent(partial, collector.agent_names())
new_trace = collector.step(partial, selected)

print(f"Orchestrator chose {selected!r} → new trace:", new_trace)
```

**Key calls**

| Call | What it does |
|------|--------------|
| `Orchestrator.select_agent(partial_trace, agent_names)` | Returns the name of the agent the policy thinks is best for the current partial trace. |
| `TraceCollector.step(partial_trace, agent_name)` | Executes the chosen agent and returns an updated trace dictionary. |
| `Orchestrator.forward(partial_trace)` | Returns the raw logits for each candidate agent (used internally by `select_agent`). |

---

## 5️⃣ Saving & loading the trained