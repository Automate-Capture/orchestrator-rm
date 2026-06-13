from collections import defaultdict
from typing import List, Tuple

from orchestrator_rm.cost_metric import CostMetric


class PairGenerator:
    """Clusters traces by outcome, pairs within clusters, assigns winner by cost."""

    def __init__(self, cost_metric: CostMetric = None):
        self.cost_metric = cost_metric or CostMetric()

    def generate_pairs(self, traces: List[dict]) -> List[Tuple[dict, dict]]:
        clusters = self._cluster_by_outcome(traces)
        pairs: List[Tuple[dict, dict]] = []
        for cluster_traces in clusters.values():
            pairs.extend(self._pair_within_cluster(cluster_traces))
        return pairs

    def _cluster_by_outcome(self, traces: List[dict]) -> dict:
        clusters: dict = defaultdict(list)
        for trace in traces:
            outcome = trace.get("outcome", "")
            clusters[outcome].append(trace)
        return clusters

    def _pair_within_cluster(self, traces: List[dict]) -> List[Tuple[dict, dict]]:
        if len(traces) < 2:
            return []

        scored = [(self.cost_metric.cost(t), t) for t in traces]
        scored.sort(key=lambda x: x[0])

        pairs: List[Tuple[dict, dict]] = []
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                cost_i, trace_i = scored[i]
                cost_j, trace_j = scored[j]
                if cost_i < cost_j:
                    pairs.append((trace_i, trace_j))
        return pairs
