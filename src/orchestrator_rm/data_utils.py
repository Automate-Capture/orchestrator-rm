import random
from typing import List

from orchestrator_rm.trace_collector import TraceCollector


def _make_calculator_handler(correct_answers):
    def handler(query, ctx):
        rng = ctx["rng"]
        tokens = rng.randint(10, 20)
        answer = correct_answers.get(query, "unknown")
        return {
            "tokens": tokens,
            "output_tokens": rng.randint(5, 10),
            "content": str(answer),
            "tool": "compute",
        }
    return handler


def _make_planner_handler():
    def handler(query, ctx):
        rng = ctx["rng"]
        return {
            "tokens": rng.randint(30, 50),
            "output_tokens": rng.randint(15, 25),
            "content": f"Plan: solve '{query}' step by step",
            "tool": "plan",
        }
    return handler


def _make_verifier_handler(correct_answers):
    def handler(query, ctx):
        rng = ctx["rng"]
        answer = correct_answers.get(query, "unknown")
        return {
            "tokens": rng.randint(15, 25),
            "output_tokens": rng.randint(5, 10),
            "content": str(answer),
            "tool": "verify",
        }
    return handler


def _make_searcher_handler(correct_answers):
    def handler(query, ctx):
        rng = ctx["rng"]
        answer = correct_answers.get(query, "unknown")
        return {
            "tokens": rng.randint(80, 150),
            "output_tokens": rng.randint(30, 50),
            "content": str(answer),
            "tool": "search",
        }
    return handler


def _make_summarizer_handler():
    def handler(query, ctx):
        rng = ctx["rng"]
        prev = ctx.get("context", {}).get("results", [])
        content = prev[-1].get("content", "") if prev else query
        return {
            "tokens": rng.randint(40, 70),
            "output_tokens": rng.randint(20, 35),
            "content": content,
            "tool": "summarize",
        }
    return handler


AGENT_NAMES = ["calculator", "planner", "verifier", "searcher", "summarizer"]

STRATEGIES = {
    "efficient": ["calculator"],
    "moderate": ["calculator", "verifier"],
    "thorough": ["planner", "calculator", "verifier"],
    "wasteful": ["searcher", "planner", "calculator", "verifier", "summarizer"],
}


class SyntheticDataset:
    """Generates synthetic arithmetic queries and deterministic agent traces."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._queries: List[str] = []
        self._answers: dict = {}
        self._collector: TraceCollector = None

    @property
    def queries(self) -> List[str]:
        return self._queries

    @property
    def collector(self) -> TraceCollector:
        if self._collector is None:
            raise RuntimeError("Call make_dataset() or make_contrastive_pair() first")
        return self._collector

    @property
    def agent_names(self) -> List[str]:
        return list(AGENT_NAMES)

    def _generate_queries(self, n: int):
        queries = []
        answers = {}
        for _ in range(n):
            a = self.rng.randint(1, 100)
            b = self.rng.randint(1, 100)
            op = self.rng.choice(["+", "-", "*"])
            query = f"Compute: {a} {op} {b}"
            if op == "+":
                answer = a + b
            elif op == "-":
                answer = a - b
            else:
                answer = a * b
            queries.append(query)
            answers[query] = answer
        return queries, answers

    def _build_collector(self, answers: dict) -> TraceCollector:
        collector = TraceCollector()
        collector.register_agent("calculator", _make_calculator_handler(answers))
        collector.register_agent("planner", _make_planner_handler())
        collector.register_agent("verifier", _make_verifier_handler(answers))
        collector.register_agent("searcher", _make_searcher_handler(answers))
        collector.register_agent("summarizer", _make_summarizer_handler())
        self._collector = collector
        return collector

    def make_dataset(self, num_queries: int = 50, traces_per_query: int = 4) -> List[dict]:
        queries, answers = self._generate_queries(num_queries)
        self._queries = queries
        self._answers = answers

        collector = self._build_collector(answers)
        strategy_names = list(STRATEGIES.keys())
        traces = []

        for query in queries:
            for j in range(min(traces_per_query, len(strategy_names))):
                strategy = STRATEGIES[strategy_names[j]]
                seed = self.rng.randint(0, 2**31)
                trace = collector.collect(query, strategy, seed=seed)
                trace["strategy"] = strategy_names[j]
                traces.append(trace)

        return traces

    def make_contrastive_pair(self) -> tuple:
        """Hand-crafted efficient vs inefficient pair with identical outcome."""
        query = "Compute: 42 + 58"
        answers = {query: 100}
        collector = self._build_collector(answers)

        efficient = collector.collect(query, ["calculator"], seed=0)
        inefficient = collector.collect(
            query,
            ["searcher", "planner", "calculator", "verifier", "summarizer"],
            seed=0,
        )

        return efficient, inefficient
