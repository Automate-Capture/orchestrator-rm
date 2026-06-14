# Architecture

OrchestrateRM is organized around a small set of modules that form a pipeline:

```mermaid
flowchart LR
    A["Synthetic traces"]
    B["Pair generation"]
    C["Reward model"]
    D["Evaluation"]
    A --> B
    B --> C
    C --> D
```

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
