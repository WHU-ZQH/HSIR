# H-GRPO

H-GRPO augments GRPO with the intrinsic diversity reward described in
[HSIR](https://arxiv.org/abs/2605.24998), Eq. 4-5 and Appendix B.
It supports MedQA and GSM8K.

## Method

The rollout policy supplies middle-layer hidden states and self-attention for
each sampled solution. With H shaped [features, tokens], the scorer computes
H.T @ J @ H and its singular values. Attention-weighted log singular values
give the raw diversity score. Scores are standardized within each query's
sampled group (Eq. 5), then divided by that group's maximum standardized score
(Eq. 9). The final reward is:

```text
R = R_accuracy + R_format + 0.2 * R_InDiv
```

Groups are gathered across processes before normalization. Constant-score groups
receive zero diversity reward. Below-mean scores can receive negative rewards;
values are not clipped to [0, 1]. The policy loss is normalized by valid tokens
across training processes.

### Scoring conventions

- Score all valid prompt and completion tokens; exclude padding.
- For N blocks, use output of block max(1, N // 2) and its self-attention.
- For each key token, maximize attention across heads and valid query positions,
  then normalize token weights to sum to one.
- Pair token weights in sequence order with descending singular values.
- Add 1e-3 * I to the covariance for stability. Use float32 scoring with bfloat16
  models; float64 inputs remain float64.
- Request eager attention and score one sequence per forward to bound memory.
  Use the KL reference model only for the KL term.

## Environment

Use a separate environment from LLaMA-Factory. This trainer uses the TRL API
with GRPOConfig.vllm_device, trl.extras.profiling and
trl.trainer.utils.selective_log_softmax. Select a compatible
TRL/Transformers/vLLM/Accelerate stack before GPU training; an exact validated
environment lock is not included.

Also install datasets, deepspeed, peft, safetensors, numpy, math-verify,
latex2sympy2-extended, packaging and PyYAML. W&B is optional with reporting off.
The custom trainer is imported locally.

## Training

From this directory:

```bash
bash scripts/train.sh MedQA /models/seed ../data/MedQA/medqa_grpo_unlabeled.json outputs/medqa
bash scripts/train.sh GSM8K /models/seed ../data/GSM8K/gsm8k_grpo_unlabeled.json outputs/gsm8k
```

Recipes use learning rate 1e-6, 300 optimizer steps, 8 generations per query,
and an effective batch of 14 queries. The default launcher allocates GPUs 0-7:
seven training processes and one vLLM device. Each training process handles
16 sequences with one accumulation step:

```text
7 processes * 16 sequences * 1 accumulation / 8 generations = 14 queries/update
```

Set CUDA_VISIBLE_DEVICES, NUM_PROCESSES and LEARNING_RATE as needed. Changing
process count or recipe batch settings changes the effective batch size. The
global microbatch must be divisible by num_generations. Batch size is counted
in queries; generated sequences are counted separately. Memory requirements
depend on the model and sequence length.

Use a fresh output directory. Remote reporting and Hub upload are disabled.
Local JSON uses instruction, optional input, and output; see
[data documentation](../docs/DATA.md). The registry key indiv_score_medical
selects the same diversity reward for both tasks and may appear anywhere in
the reward list.

## Validation

From the repository root:

```bash
pip install torch
python -m unittest discover -s tests -p test_hgrpo.py -v
```

CPU tests cover covariance, attention weighting, padding, rank deficiency,
group normalization, distributed reward assembly, middle-layer selection and
token-loss normalization. GPU training and full benchmark reproduction have
not been run for this release.

Original Open-R1/TRL license headers are retained. See [LICENSE](../LICENSE)
and [third-party notices](../THIRD_PARTY_NOTICES.md).
