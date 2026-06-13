"""Self-supervised Bradley-Terry reward model for multi-agent orchestration."""

__version__ = "0.1.0"

from .orchestrator import Orchestrator
from .reward_model import OrchestratorRewardModel

__all__ = ["Orchestrator", "OrchestratorRewardModel"]
