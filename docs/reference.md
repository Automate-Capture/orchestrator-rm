# OrchestrateRM – API Reference

*Version: 0.1.0*  
*Package name:* `orchestrator_rm`  

All public symbols are imported from the `orchestrator_rm` package, e.g.:

```python
from orchestrator_rm import reward_model, orchestrator, encoder
```

Below you will find a concise reference for every public class and function that ships with the library.  
For each entry we show:

* **Exact signature** (including type hints)  
* **Short description** – what the object does and any important notes  
* **Example usage** – a minimal, runnable snippet that demonstrates the typical way to call it.

---

## `src/orchestrator_rm/cost_metric.py`

### `cost(self, trace: dict) -> float`
* **Description** – Computes a scalar cost for a single trace dictionary. The implementation follows the “hard part” of the self‑supervised Bradley‑Terry reward model: lower cost → higher quality. The method is meant to be mixed into a class that stores the cost‑related hyper‑parameters (e.g., weighting of latency vs. token usage).  
* **Example**

```python
from orchestrator_rm.cost_metric import CostMetric

class MyMetric(CostMetric):
    def __init__(self, latency_weight: float = 0.5):
        self.latency_weight = latency_weight

metric = MyMetric()
trace = {"latency_ms": 120, "tokens_used": 350}
cost = metric.cost(trace)          # → float
print(cost)
```

---

## `src/orchestrator_rm/data_utils.py`

### `handler(query, ctx)`
* **Description** – A generic data‑handler stub used by the test harness. It receives a `query` string and a context dictionary `ctx`, mutates `ctx` (e.g., storing intermediate results), and returns a possibly transformed query.  
* **Example**

```python
from orchestrator_rm.data_utils import handler

def my_handler(query, ctx):
    ctx["length"] = len(query)
    return query.lower()

ctx = {}
new_query = handler("Hello World", ctx)
print(new_query)   # "hello world"
print(ctx)         # {"length": 11}
```

*(The same signature appears five times in the source; each occurrence is a separate handler used for different stages of the pipeline.)*

### `queries(self) -> List[str]`
* **Description** – Returns the list of queries that the dataset generator will use. Typically called on a `DatasetBuilder`‑like object.  
* **Example**

```python
from orchestrator_rm.data_utils import DatasetBuilder

builder = DatasetBuilder()
print(builder.queries())   # ["What is AI?", "Explain quantum computing", ...]
```

### `collector(self) -> TraceCollector`
* **Description** – Instantiates (or returns a cached) `TraceCollector` that will be used to gather execution traces for the current dataset.  
* **Example**

```python
collector = builder.collector()
print(type(collector))   # <class orchestrator_rm.trace_collector.TraceCollector>
```

### `agent_names(self) -> List[str]`
* **Description** – Returns the names of all agents that have been registered with the collector.  
* **Example**

```python
print(builder.agent_names())   # ["gpt-4", "claude-2", "llama-2"]
```

### `make_dataset(self, num_queries: int = 50, traces_per_query: int = 4) -> List[dict]`
* **Description** – Generates a synthetic dataset of `num_queries` queries, each with `traces_per_query` collected traces. The returned list contains dictionaries with keys `query`, `trace`, and optional metadata.  
* **Example**

```python
dataset = builder.make_dataset(num_queries=10, traces_per_query=3)
print(len(dataset))   # 30
print(dataset[0].keys())   # dict_keys(['query', 'trace', 'metadata'])
```

### `make_contrastive_pair(self) -> tuple`
* **Description** – Randomly selects two traces belonging to the same query and returns them as a tuple `(trace_a, trace_b)`. Used for contrastive (Bradley‑Terry) training.  
* **Example**

```python
pair = builder.make_contrastive_pair()
trace_a, trace_b = pair
print(trace_a["trace_id"], trace_b["trace_id"])
```

---

## `src/orchestrator_rm/encoder.py`

### `class PositionalEncoding(nn.Module)`
* **Description** – Standard sinusoidal positional encoding module (as in the original Transformer paper). Adds positional information to token embeddings.  
* **Signature**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

* **Example**

```python
import torch
from orchestrator_rm.encoder import PositionalEncoding

pos_enc = PositionalEncoding(d_model=128)
emb = torch.randn(10, 32, 128)          # (seq_len, batch, d_model)
emb_pe = pos_enc(emb)                   # same shape, with positional info added
```

### `def forward(self, x: torch.Tensor) -> torch.Tensor`
* **Description** – Implements the forward pass for `PositionalEncoding`. Returns `x + positional_encoding`.  
* **Example** – see above.

### `class TraceEncoder(nn.Module)`
* **Description** – Encodes a trace dictionary into a fixed‑size vector. Internally it tokenizes the trace, embeds tokens, adds positional encodings, and runs a small transformer encoder.  
* **Signature**

```python
class TraceEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2)
    def get_agent_id(self, agent_name: str) -> int
    def tokenize_trace(self, trace: dict) -> dict
    def encode(self, trace: dict) -> torch.Tensor
    def encode_batch(self, traces: List[dict]) -> torch.Tensor
```

* **Example**

```python
from orchestrator_rm.encoder import TraceEncoder

encoder = TraceEncoder(vocab_size=10000)
trace = {"agent": "gpt-4", "output": "Hello", "latency_ms": 120}
vec = encoder.encode(trace)          # shape: (d_model,)
print(vec.shape)
```

#### `def get_agent_id(self, agent_name: str) -> int`
* **Description** – Maps an agent name to a unique integer ID used in the tokenization scheme.  
* **Example**

```python
agent_id = encoder.get_agent_id("gpt-4")   # e.g., 3
```

#### `def tokenize_trace(self, trace: dict) -> dict`
* **Description** – Converts a trace dict into a tokenized representation (e.g., token IDs for strings, quantized values for numbers). Returns a dict with fields `input_ids`, `attention_mask`, etc.  
* **Example**

```python
tokens = encoder.tokenize_trace(trace)
print(tokens["input_ids"][:5])
```

#### `def encode(self, trace: dict) -> torch.Tensor`
* **Description** – Tokenizes a single trace and returns its embedding vector (`torch.Tensor` of shape `(d_model,)`).  
* **Example** – see `TraceEncoder` usage above.

#### `def encode_batch(self, traces: List[dict]) -> torch.Tensor`
* **Description** – Encodes a list of traces in a single forward pass. Returns a tensor of shape `(batch, d_model)`.  
* **Example**

```python
batch_vec = encoder.encode_batch([trace, trace])
print(batch_vec.shape)   # torch.Size([2, 128])
```

---

## `src/orchestrator_rm/eval.py`

### `def evaluate_pairwise(self, trace_a: dict, trace_b: dict) -> dict`
* **Description** – Given two traces for the same query, computes a Bradley‑Terry style probability that `trace_a` is preferred over `trace_b`. Returns a dictionary containing the raw scores, the probability, and optional diagnostics.  
* **Example**

```python
from orchestrator_rm.eval import Evaluator

evaluator = Evaluator()
result = evaluator.evaluate_pairwise(trace_a, trace_b)
print(result["prob_a_wins"])   # 0.73
```

### `def run_acceptance_test(self) -> bool`
* **Description** – Executes a lightweight sanity‑check on the reward model (e.g., verifies that higher‑quality traces receive higher scores). Returns `True` if the model passes the test, otherwise `False`.  
* **Example**

```python
if evaluator.run_acceptance_test():
    print("Reward model looks healthy!")
else:
    print("Something is wrong with the model.")
```

---

## `src/orchestrator_rm/orchestrator.py`

### `class Orchestrator(nn.Module)`
* **Description** – A neural policy that, given a partial execution trace, predicts the next agent to invoke. Internally it uses the `TraceEncoder` and a small decision