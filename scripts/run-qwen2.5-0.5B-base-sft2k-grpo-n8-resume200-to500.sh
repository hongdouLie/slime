#!/bin/bash
# Minimal GRPO smoke test for slime on NVIDIA DGX Spark (GB10, single GPU).
# Goal: exercise the full rollout → reward → policy-update loop for one tiny
# step and exit cleanly. Used only to validate the GB10 port; not a training
# recipe.
#
# Prerequisites:
#   - /root/Qwen2.5-0.5B-Instruct                    (HF checkpoint)
#   - /root/Qwen2.5-0.5B-Instruct_torch_dist         (from tools/convert_hf_to_torch_dist.py)
#   - /data3/private/lwz/datasets/slime_gsm8k/train.parquet

set -ex

# clean any leftover ray/sglang

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /data3/private/lwz/models/Qwen2.5-0.5B/
   --ref-load /data3/private/lwz/models/Qwen2.5-0.5B_torch_dist/
   --load /data3/private/lwz/slime-runs/qwen25_05b_base_sft2k_grpo_n8_curve200/
   --save /data3/private/lwz/slime-runs/qwen25_05b_base_sft2k_grpo_n8_curve500/
   --save-interval 25
)

ROLLOUT_ARGS=(
   --prompt-data /data3/private/lwz/datasets/slime_gsm8k/train.parquet
   --rollout-shuffle
   --input-key messages
   --label-key label
   --apply-chat-template
   --rm-type math

   --num-rollout 500
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --num-steps-per-rollout 1
   --global-batch-size 32

   --rollout-max-response-len 768
   --rollout-stop-token-ids 151645 151643
   --rollout-temperature 1
   --rollout-top-p 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --override-opt-param-scheduler
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.35
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ray start --head --temp-dir /data3/private/lwz/slime-stack/ray-tmp --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/data3/private/lwz/slime-stack/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
