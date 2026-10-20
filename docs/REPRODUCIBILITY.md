# Experiment configuration and validation

Paper: ICML 2026 (arXiv:2605.24998v1). This release provides iterative SFT/DPO and H-GRPO
for MedQA and GSM8K. Model checkpoints and historical training outputs are
stored separately.

## Iterative SFT/DPO

For main Qwen2.5 runs: K=10, J=5, tau=-0.5, T=3; SFT batch 32, DPO batch 16,
three epochs, cutoff 2048. Smaller model learning rates are 1e-5/1e-6;
7B/8B rates are 2e-6/2e-7 for SFT/DPO. Additional model studies use one iteration
where specified. The runner exposes iteration count and learning rates.
SFT starts from the base model; DPO continues from the previous model.

The pipeline preserves rejected responses for DPO, bounds candidate counts,
handles constant-score groups, and registers generated datasets with
LLaMA-Factory. Optional preference seeds must contain chosen/rejected pairs.
See [DATA.md](DATA.md) for input contracts.

## H-GRPO

Rewards combine accuracy, format and attention-weighted intrinsic diversity
with weights 1, 1 and 0.2. Normalization operates on each query's eight
generations after gathering scores across processes. Defaults use 14 queries
per optimizer update, learning rate 1e-6 and 300 steps.
See [H-GRPO README](../H-GRPO/README.md) for scoring conventions, batch-size
calculation and launch commands.

## Validation scope

Validation covers Python syntax, synthetic CPU regression tests, numerical
H-GRPO tests and training-plan generation. GPU training, GPU inference and
full benchmark reproduction have not been performed for this release.
Validate the CUDA/library stack before a full run; dependency ranges are not
a tested environment lock.
