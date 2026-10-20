# HSIR

**Better, Faster: Harnessing Self-Improvement in Large Reasoning Models**

Qihuang Zhong, Liang Ding, Juhua Liu, Bo Du, Leszek Rutkowski, and Dacheng Tao.

[Paper](https://arxiv.org/abs/2605.24998) · [Data format](docs/DATA.md) ·
[Reproducibility notes](docs/REPRODUCIBILITY.md) · [中文说明](README_zh.md)

HSIR improves reasoning using self-generated candidates, verify-then-exit sampling,
and hidden-state-based selection. This repository includes the MedQA and GSM8K
SFT/DPO implementations and a separate H-GRPO implementation.

## Installation

Use Linux, NVIDIA GPUs and Python 3.10 or 3.11. The paper used A800 GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Choose a compatible CUDA/PyTorch/vLLM stack. The dependency ranges are **not a
verified environment lock**. Hidden-state scoring uses eager attention and can
require substantial GPU memory.

Install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), with DeepSpeed,
in its own environment and pass its checkout and interpreter to the runner.
The retained training requirements used Transformers 4.49–4.55, TRL 0.8.6–0.9.6
and PEFT up to 0.15.2; those older training APIs should be isolated from vLLM.
H-GRPO requires a third environment; see [its README](H-GRPO/README.md).

## Data

Provide a seed SFT dataset and an unlabeled-for-training pool with reference
answers retained for verification. Both are JSON arrays. MedQA uses Alpaca-style
`instruction`/`input`/`system`/`output`; GSM8K uses `problem`/`system` and the
historical `groud_truth_solution` field. Full schemas and synthetic examples are
in [docs/DATA.md](docs/DATA.md). Basic MedQA and GSM8K datasets are bundled
under `data/`: seed SFT, training pools, GRPO inputs and test sets. Model weights
are supplied separately. Record counts and checksums are in `data/manifest.json`.

## Training

Preview the pipeline without GPUs or file writes:

```bash
python run.py --task MedQA --mode sft \
  --base-model /models/Qwen2.5-1.5B-Instruct \
  --seed-data data/MedQA/medqa_train_ds_seed.json \
  --train-data data/MedQA/medqa_train_ds_unlabeled.json \
  --llamafactory /opt/LLaMA-Factory \
  --train-python /opt/lf-env/bin/python \
  --output outputs/medqa-hsir-sft \
  --gpus 0,1,2,3,4,5,6,7 --iterations 3 --dry-run
```

Remove `--dry-run` to train. Use `--task GSM8K` with the corresponding data files,
or `--mode dpo` with a fresh output path. The runner performs seed SFT, rejection
sampling (K=10), verify-then-exit sampling (J=5), correctness separation,
hidden-state scoring, threshold filtering (−0.5), and iterative training.
Original wrong answers are retained for preference pairs.

Each SFT iteration starts from the base model and mixes seed examples with
selected generated examples. DPO continues from the previous model. Ordinary
SFT seed examples cannot be passed as DPO pairs: use `--preference-seed` only
if a matching `chosen`/`rejected` seed dataset is available; otherwise DPO trains
on generated pairs. This resolves an incompatible dataset mixture in the old
launcher and must be considered when comparing historical experiments.

Default learning rates are `--sft-lr 1e-5 --dpo-lr 1e-6`. For the paper's 7B/8B
runs specify `--sft-lr 2e-6 --dpo-lr 2e-7`. Global batch sizes are 32 (SFT) and
16 (DPO), with three epochs, cutoff 2048, DPO beta 0.1 and auxiliary SFT weight
0.5. Use `--iterations 1` for experiments reported with one iteration.
Select the model's LLaMA-Factory template via `--template` (default `qwen`).

Artifacts are grouped into `models/`, `prepared_data/` and `iteration_N/`.
`run.json` records parameters; training dataset registrations are generated
automatically. Failed worker processes stop the run before merging outputs.
Existing nonempty run directories are refused; automated pipeline resume is
not implemented. Use a fresh directory after inspecting an interrupted run.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python src/MedQA/step8_model_inference.py \
  outputs/medqa-hsir-sft/models/hsir_sft_3 \
  /datasets/medqa_test.json outputs/medqa-eval 0 0 1
python src/MedQA/merge_evaluation.py outputs/medqa-eval test_predictions
```

For GSM8K, replace both `MedQA` path components with `GSM8K` and use its test
schema. Positional inference arguments are model, test JSON, output directory,
temperature, zero-based shard index and shard count. Temperature 0 uses greedy
decoding with a 4096-token output limit. The merger reports accuracy and saves
predictions. See the data guide for reference-answer formatting.

## H-GRPO

[H-GRPO](H-GRPO/README.md) combines accuracy, format and attention-weighted
intrinsic diversity rewards, using a local modified TRL trainer. Its recipes
use 8 generations per query, 14 queries per update and 300 optimizer steps.
See its README for installation, launch commands and CPU validation.

## Tests

```bash
pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

Tests cover synthetic candidate preservation, score normalization, dataset
preparation and dry-run planning. No full GPU experiments were executed for
this code preparation.

## Citation and license

See [citation.bib](citation.bib). Code is licensed under [Apache-2.0](LICENSE).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for Open-R1/TRL attribution.
