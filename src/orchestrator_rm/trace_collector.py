import random
from typing import Callable, Dict, List, Optional


AgentHandler = Callable[[str, dict], dict]


class TraceCollector:
    """Runs queries through a mock multi-agent pipeline, logging every decision."""

    def __init__(self):
        self._agents: Dict[str, AgentHandler] = {}

    def register_agent(self, name: str, handler: AgentHandler):
        self._agents[name] = handler

    @property
    def agent_names(self) -> List[str]:
        return list(self._agents.keys())

    def collect(self, query: str, strategy: List[str], seed: Optional[int] = None) -> dict:
        rng = random.Random(seed)
        events: List[dict] = []
        step = 0
        context: dict = {"query": query, "results": []}

        for agent_name in strategy:
            if agent_name not in self._agents:
                raise ValueError(f"Unknown agent: {agent_name}")

            events.append({
                "type": "select",
                "agent": agent_name,
                "tokens": 0,
                "step": step,
            })
            step += 1

            handler = self._agents[agent_name]
            result = handler(query, {"rng": rng, "context": context})

            events.append({
                "type": "tool_call",
                "agent": agent_name,
                "tokens": result.get("tokens", 0),
                "step": step,
                "tool": result.get("tool", agent_name),
            })
            step += 1

            events.append({
                "type": "result",
                "agent": agent_name,
                "tokens": result.get("output_tokens", 0),
                "step": step,
                "content": result.get("content", ""),
            })
            context["results"].append(result)
            step += 1

        total_tokens = sum(e.get("tokens", 0) for e in events)
        total_calls = sum(1 for e in events if e["type"] == "tool_call")
        outcome = context["results"][-1].get("content", "") if context["results"] else ""

        return {
            "query": query,
            "events": events,
            "outcome": outcome,
            "metadata": {
                "total_tokens": total_tokens,
                "total_calls": total_calls,
            },
        }

    def step(self, partial_trace: dict, agent_name: str, rng=None) -> dict:
        if rng is None:
            rng = random.Random()
        if agent_name not in self._agents:
            raise ValueError(f"Unknown agent: {agent_name}")

        trace = {
            "query": partial_trace["query"],
            "events": list(partial_trace.get("events", [])),
            "outcome": partial_trace.get("outcome", ""),
            "metadata": {
                "total_tokens": partial_trace.get("metadata", {}).get("total_tokens", 0),
                "total_calls": partial_trace.get("metadata", {}).get("total_calls", 0),
            },
        }

        step_num = len(trace["events"])
        handler = self._agents[agent_name]
        context = {"query": trace["query"], "results": []}
        result = handler(trace["query"], {"rng": rng, "context": context})

        trace["events"].extend([
            {"type": "select", "agent": agent_name, "tokens": 0, "step": step_num},
            {
                "type": "tool_call", "agent": agent_name,
                "tokens": result.get("tokens", 0),
                "step": step_num + 1, "tool": result.get("tool", agent_name),
            },
            {
                "type": "result", "agent": agent_name,
                "tokens": result.get("output_tokens", 0),
                "step": step_num + 2, "content": result.get("content", ""),
            },
        ])

        trace["outcome"] = result.get("content", "")
        trace["metadata"]["total_tokens"] += result.get("tokens", 0) + result.get("output_tokens", 0)
        trace["metadata"]["total_calls"] += 1

        return trace
