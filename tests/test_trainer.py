import torch

from orchestrator_rm.trainer import OrchestrationTrainer


def test_train_end_to_end():
    torch.manual_seed(42)
    trainer = OrchestrationTrainer(d_model=16, nhead=4, num_layers=1, seed=42)
    result = trainer.train(
        num_queries=10,
        rm_epochs=5,
        policy_epochs=2,
        rm_lr=1e-3,
        policy_lr=1e-4,
    )

    assert result["num_traces"] == 40  # 10 queries × 4 strategies
    assert result["num_pairs"] > 0
    assert len(result["rm_losses"]) == 5
    assert len(result["policy_losses"]) == 2
    assert trainer.reward_model is not None
    assert trainer.orchestrator is not None


def test_rm_loss_decreases():
    torch.manual_seed(42)
    trainer = OrchestrationTrainer(d_model=16, nhead=4, num_layers=1, seed=42)
    result = trainer.train(num_queries=15, rm_epochs=15, policy_epochs=0)
    losses = result["rm_losses"]
    assert losses[-1] < losses[0]
