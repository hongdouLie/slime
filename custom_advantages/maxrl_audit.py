import json
import torch

from custom_advantages.maxrl import (
    compute_maxrl_advantages as compute_original_maxrl_advantages,
)

_CALL_COUNT = 0


def _tolist(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return list(value)


def compute_maxrl_advantages(args, rollout_data):
    global _CALL_COUNT

    compute_original_maxrl_advantages(args, rollout_data)

    rewards = torch.as_tensor(
        rollout_data["rewards"],
        dtype=torch.float32,
    ).detach().cpu()

    group_size = int(args.n_samples_per_prompt)
    grouped_rewards = rewards.reshape(-1, group_size)
    pass_rates = grouped_rewards.mean(dim=1)

    advantages = rollout_data["advantages"]
    sequence_advantages = torch.tensor(
        [
            float(adv.detach().float().reshape(-1)[0].cpu())
            if adv.numel() > 0 else float("nan")
            for adv in advantages
        ],
        dtype=torch.float32,
    )
    grouped_advantages = sequence_advantages.reshape(-1, group_size)

    token_lengths = [
        int(adv.numel())
        for adv in advantages
    ]

    unique_rewards = sorted(set(float(x) for x in rewards.tolist()))
    binary_rewards = all(x in (0.0, 1.0) for x in unique_rewards)

    report = {
        "audit_call": _CALL_COUNT,
        "n_samples_per_prompt": group_size,
        "num_sequences": int(rewards.numel()),
        "num_groups": int(grouped_rewards.shape[0]),
        "unique_rewards": unique_rewards,
        "binary_rewards": binary_rewards,
        "grouped_rewards": grouped_rewards.tolist(),
        "pass_rates": pass_rates.tolist(),
        "grouped_advantages": grouped_advantages.tolist(),
        "advantage_group_means": grouped_advantages.mean(dim=1).tolist(),
        "token_lengths": token_lengths,
        "total_response_tokens": sum(token_lengths),
        "normalize_advantages": bool(
            getattr(args, "normalize_advantages", False)
        ),
        "calculate_per_token_loss": bool(
            getattr(args, "calculate_per_token_loss", False)
        ),
    }

    print(
        "MAXRL_AUDIT " + json.dumps(report, ensure_ascii=False),
        flush=True,
    )

    if not binary_rewards:
        raise RuntimeError(
            "MaxRL audit failed: rewards are not raw binary rewards"
        )

    if not torch.allclose(
        grouped_advantages.mean(dim=1),
        torch.zeros(grouped_advantages.shape[0]),
        atol=1e-5,
        rtol=0,
    ):
        raise RuntimeError(
            "MaxRL audit failed: group advantage mean is not zero"
        )

    _CALL_COUNT += 1
