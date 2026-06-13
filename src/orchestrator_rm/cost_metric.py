class CostMetric:
    """Computes scalar efficiency cost from a trace: tokens + call penalty."""

    def __init__(self, token_weight: float = 1.0, call_penalty: float = 100.0):
        self.token_weight = token_weight
        self.call_penalty = call_penalty

    def cost(self, trace: dict) -> float:
        metadata = trace.get("metadata", {})
        total_tokens = metadata.get("total_tokens", 0)
        total_calls = metadata.get("total_calls", 0)
        return self.token_weight * total_tokens + self.call_penalty * total_calls
