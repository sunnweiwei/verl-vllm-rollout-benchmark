import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict


def test_bypass_mode_requires_and_uses_rollout_log_probs():
    from verl.protocol import DataProto
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

    response_mask = torch.ones(2, 3)
    rollout_log_probs = torch.tensor([[-1.0, -1.1, -1.2], [-2.0, -2.1, -2.2]])
    batch = DataProto(
        batch=TensorDict(
            {
                "response_mask": response_mask,
                "rollout_log_probs": rollout_log_probs,
            },
            batch_size=[2],
        )
    )
    policy_loss_config = OmegaConf.create({"loss_mode": "ppo_clip"})

    apply_bypass_mode(batch, RolloutCorrectionConfig(), policy_loss_config)

    assert torch.equal(batch.batch["old_log_probs"], rollout_log_probs)
    assert policy_loss_config.loss_mode == "bypass_mode"


def test_bypass_mode_fails_loudly_without_rollout_log_probs():
    from verl.protocol import DataProto
    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

    batch = DataProto(batch=TensorDict({"response_mask": torch.ones(1, 2)}, batch_size=[1]))

    with pytest.raises(ValueError, match="requires rollout_log_probs"):
        apply_bypass_mode(batch, policy_loss_config=OmegaConf.create({"loss_mode": "ppo_clip"}))


def test_rollout_correction_weights_match_response_mask_shape():
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    old_log_prob = torch.tensor([[-1.0, -2.0, -3.0]])
    rollout_log_prob = torch.tensor([[-1.1, -1.9, -3.2]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0]])

    weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is="token",
        rollout_is_threshold=10.0,
        rollout_rs=None,
    )

    assert weights_proto.batch["rollout_is_weights"].shape == response_mask.shape
    assert torch.equal(modified_mask, response_mask)
    assert "rollout_corr/rollout_is_mean" in metrics


def test_router_replay_missing_routed_experts_is_a_plumbing_error():
    from verl.utils.veomni.router_replay import VeOmniRouterReplay

    controller = VeOmniRouterReplay()
    controller.begin_replay()

    with pytest.raises(RuntimeError, match="has no target"):
        controller.on_router_forward(
            module=object(),
            routing_scores=torch.randn(3, 4),
            top_indices=torch.tensor([[0], [1], [2]]),
        )


def test_router_replay_consumes_routed_experts_when_present():
    from verl.utils.veomni.router_replay import VeOmniRouterReplay

    controller = VeOmniRouterReplay()
    controller.begin_replay()
    target = torch.tensor([[2], [1], [0]])
    native = torch.tensor([[0], [0], [0]])
    controller.set_microbatch_targets([target])

    replayed = controller.on_router_forward(
        module=object(),
        routing_scores=torch.randn(3, 4),
        top_indices=native,
    )

    assert torch.equal(replayed, target)
