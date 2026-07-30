import torch


def compute_maxrl_sequence_advantages(
    rewards: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute MaxRL sequence-level advantages."""
    if rewards.ndim != 1:
        raise ValueError(
            f"rewards must be 1-D, got shape={tuple(rewards.shape)}"
        )

    if group_size <= 0:
        raise ValueError(
            f"group_size must be positive, got {group_size}"
        )

    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"reward count {rewards.numel()} is not divisible by "
            f"group_size {group_size}"
        )

    grouped_rewards = rewards.reshape(-1, group_size)
    pass_rates = grouped_rewards.mean(dim=1, keepdim=True)

    advantages = torch.where(
        pass_rates > 0,
        (grouped_rewards - pass_rates) / (pass_rates + epsilon),
        torch.zeros_like(grouped_rewards),
    )

    return advantages.reshape(-1)


def compute_maxrl_advantages(args, rollout_data) -> None:
    """Slime custom advantage hook implementing MaxRL."""
    rewards = rollout_data["rewards"]
    kl = rollout_data["kl"]

    if len(rewards) != len(kl):
        raise ValueError(
            f"reward count {len(rewards)} does not match "
            f"sequence count {len(kl)}"
        )

    device = kl[0].device
    reward_tensor = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=device,
    )

    sequence_advantages = compute_maxrl_sequence_advantages(
        rewards=reward_tensor,
        group_size=args.n_samples_per_prompt,
    )

    advantages = [
        torch.ones_like(token_kl, dtype=torch.float32)
        * sequence_advantages[i]
        for i, token_kl in enumerate(kl)
    ]

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = [
        advantage.clone() for advantage in advantages
    ]
