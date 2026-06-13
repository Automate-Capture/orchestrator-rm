# OrchestrateRM – Architecture Overview  

**Package name:** `orchestrator_rm`  
All public symbols are imported as `orchestrator_rm.<module>`. The package implements a self‑supervised Bradley‑Terry reward model that learns to rank orchestration traces, together with the surrounding infrastructure for data collection, encoding, pair generation, training and evaluation.

---

## 1. System Overview  

OrchestrateRM treats the problem of multi‑agent orchestration as a sequence‑to‑sequence decision process.  
1. **Trace collection** gathers execution traces from a set of heterogeneous agents (handlers).  
2. **Encoding** turns each trace into a dense vector using a positional‑encoding‑augmented transformer‑style encoder.  
3. **Pair generation** builds contrastive (A, B) pairs of traces for the same query.  
4. **Reward model** (Bradley‑Terry) learns a scalar quality score for any trace; the model is trained by maximizing the likelihood of the observed pairwise preferences.  
5. **Orchestrator** uses the learned reward model to perform policy‑gradient updates, selecting agents that are expected to improve the reward.  
6. **Evaluation** runs held‑out queries, computes pairwise win probabilities, and reports an acceptance test flag.

All components are loosely coupled through well‑defined Python interfaces, making the system easy to extend (new agents, alternative encoders, different cost metrics, etc.).

---

## 2. Module Dependency Diagram  

```mermaid
graph TD
    %% Core modules
    A[orchestrator_rm.orchestrator.Orchestrator] -->|uses| B[orchestrator_rm.reward_model.OrchestratorRewardModel]
    A -->|encodes via| C[orchestrator_rm.encoder.TraceEncoder]
    A -->|collects traces via| D[orchestrator_rm.trace_collector.TraceCollector]
    A -->|optimises with| E[orchestrator_rm.trainer.train]

    %% Data utilities
    F[orchestrator_rm.data_utils] -->|creates| G[orchestrator_rm.pair_generator.generate_pairs]
    F -->|provides| D
    F -->|exposes| H[orchestrator_rm.cost_metric.cost]

    %% Encoder details
    C -->|positional encoding| I[orchestrator_rm.encoder.PositionalEncoding]

    %% Reward model internals
    B -->|scores| C
    B -->|fits| J[orchestrator_rm.reward_model.fit]
    B -->|saves/loads| K[orchestrator_rm.reward_model.save/load]

    %% Evaluation
    L[orchestrator_rm.eval.evaluate_pairwise] -->|calls| B
    L -->|calls| C
    M[orchestrator_rm.eval.run_acceptance_test] -->|uses| L

    %% Pair generation
    G -->|takes| C
    G -->|outputs| N[Tuple[trace_a, trace_b]]

    %% Trainer
    E -->|updates| B
    E -->|updates| A
    E -->|samples| F

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#bfb,stroke:#333,stroke-width:2px
    style D fill:#ffb,stroke:#333,stroke-width:2px
    style E fill:#fdd,stroke:#333,stroke-width:2px
    style F fill:#ddd,stroke:#333,stroke-width:2px
    style G fill:#cdd,stroke:#333,stroke-width:2px
    style H fill:#c9c,stroke:#333,stroke-width:2px
    style I fill:#9cf,stroke:#333,stroke-width:2px
    style J fill:#9f9,stroke:#333,stroke-width:2px
    style K fill:#9f9,stroke:#333,stroke-width:2px
    style L fill:#fcc,stroke:#333,stroke-width:2px
    style M fill:#fcc,stroke:#333,stroke-width:2px
    style N fill:#eef,stroke:#333,stroke-width:2px
```

*Arrows indicate the direction of usage (caller → callee).*  

---

## 3. Module‑by‑Module Description  

| Module | Primary Classes / Functions | Role |
|--------|-----------------------------|------|
| **`src/orchestrator_rm/__init__.py`** | – | Exposes the public API (`Orchestrator`, `OrchestratorRewardModel`, utility functions) and defines the package version. |
| **`src/orchestrator_rm/cost_metric.py`** | `cost(trace: dict) -> float` | Computes a scalar “cost” (e.g., latency, resource usage) from a raw trace. The cost is added to the Bradley‑Terry score to bias the reward toward efficient executions. |
| **`src/orchestrator_rm/data_utils.py`** | `handler(query, ctx)`, `queries()`, `collector()`, `agent_names()`, `make_dataset()`, `make_contrastive_pair()` | Helper utilities for: <br>• Defining simple mock agents (`handler`). <br>• Enumerating a fixed set of test queries. <br>• Instantiating a `TraceCollector`. <br>• Building a synthetic dataset of traces. <br>• Sampling a single contrastive pair for quick sanity checks. |
| **`src/orchestrator_rm/encoder.py`** | `PositionalEncoding(nn.Module)`, `TraceEncoder(nn.Module)` (methods: `get_agent_id`, `tokenize_trace`, `encode`, `encode_batch`) | Turns a JSON‑like trace into a sequence of token IDs, adds sinusoidal positional encodings, and runs a small transformer encoder to produce a fixed‑size vector representation. |
| **`src/orchestrator_rm/eval.py`** | `evaluate_pairwise(trace_a, trace_b) -> dict`, `run_acceptance_test() -> bool` | Provides a high‑level evaluation API. `evaluate_pairwise` returns win probabilities and predicted scores; `run_acceptance_test` runs a small suite of held‑out queries and returns `True` if the model meets a predefined accuracy threshold. |
| **`src/orchestrator_rm/orchestrator.py`** | `Orchestrator(nn.Module)` (methods: `forward`, `select_agent`, `run`, `policy_gradient_step`) | The learning agent that decides which next sub‑agent to invoke given a partial trace. It queries the `TraceEncoder` for a representation, scores candidates with the reward model, and updates its policy via REINFORCE‑style gradients. |
| **`src/orchestrator_rm/pair_generator.py`** | `generate_pairs(traces: List[dict]) -> List[Tuple[dict, dict]]` | Given a collection of traces for the same query, produces all ordered pairs (A, B) where A ≠ B. These pairs are the training signal for the Bradley‑Terry model. |
| **`src/orchestrator_rm/reward_model.py`** | `OrchestratorRewardModel(nn.Module)` (methods: `forward`, `score`, `fit`, `save`, `load`) | Implements the Bradley‑Terry likelihood: for a pair (i, j) the probability that i is preferred is `σ(s_i - s_j)`. `fit` runs a few epochs of stochastic gradient descent on the generated pairs. Persistence utilities (`save`/`load`) store the model state. |
| **`src/orchestrator_rm/trace_collector.py`** | `TraceCollector` (methods: `register_agent`, `agent_names`, `collect`, `step`) | Central hub that runs agents on a query, records their outputs, and builds a full execution trace (a list of `(agent, action, observation)` entries). `collect` runs a full orchestration run; `step` advances a partial trace by invoking a single agent. |
| **`src/orchestrator_rm/trainer.py`** | `train(...)` | Orchestrates the end‑to‑end training loop: <br>1. Build a dataset (`make_dataset`). <br>2. Encode traces. <br>3. Generate contrastive pairs. <br>4. Fit the reward model. <br>5. Update the orchestrator policy. <br>6. Periodically evaluate and checkpoint. |

---

## 4. Data Flow  

1. **Query Generation** – `data_utils.queries()` yields a string query.  
2. **Trace Collection** – `TraceCollector.collect(query, strategy)` runs each registered agent (registered via `register_agent`) according to a *strategy* (list of agent names). Each step returns a dictionary describing the action and observation; the concatenated steps form a **trace** (`dict`).  
3. **Cost Augmentation** – `cost_metric.cost(trace)` computes a numeric cost; the cost is stored inside the trace dict (e.g., `trace["cost"]`).  
4. **Encoding** – `TraceEncoder.encode(trace)` → `torch.Tensor` representation. Internally: <br>• `tokenize_trace` maps agent