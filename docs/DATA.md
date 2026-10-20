# Data contracts

The bundled datasets are real project inputs stored as JSON arrays. Seed sets,
training pools, GRPO inputs and test sets remain separate. Use test files only
for evaluation. The examples later in this document are synthetic illustrations.

## Included files

| File under data/ | Records | Purpose |
|---|---:|---|
| MedQA/medqa_train_ds_seed.json | 1,000 | Seed SFT |
| MedQA/medqa_train_ds_unlabeled.json | 8,366 | Iterative training pool |
| MedQA/medqa_grpo_unlabeled.json | 8,366 | H-GRPO input |
| GSM8K/gsm8k_train_ds_seed.json | 1,000 | Seed SFT |
| GSM8K/gsm8k_train_ds_unlabeled.json | 5,926 | Iterative training pool |
| GSM8K/gsm8k_grpo_unlabeled.json | 5,926 | H-GRPO input |
| testset/medqa_testset.json | 1,273 | Evaluation |
| testset/gsm8k_testset.json | 1,319 | Evaluation |

GRPO inputs and iterative pools represent the same respective question sets
with different schemas; they are not additional independent training examples.
The manifest records checksums and record counts. The code license is
Apache-2.0; benchmark content retains its source dataset terms.

## MedQA

Seed SFT rows: `instruction`, `input` (empty string allowed), `system`, `output`.
Pool rows use `instruction`, optional `input`, `system`, and `label`.
The bundled pool provides the A/B/C/D reference in `label`. If using `output`
instead of `label`, it must contain `##Answer:\nA` (or B/C/D) so
verification can extract the reference label. A plain A/B/C/D label is accepted
for **evaluation** rows, where `output` is compared directly to the extracted
prediction. Do not interchange the pool and test reference formats.

```json
[{"instruction":"Synthetic format example: choose A. A: first; B: second; C: third; D: fourth.","input":"","system":"Respond with reasoning and an answer in the requested format.","output":"<think>The requested option is A.</think><answer>##Explanation:\nOption A.\n\n##Answer:\nA</answer>"}]
```

Optional `label` in the pool can supply the A/B/C/D reference directly. Input
context is retained for sampling, scoring, SFT and DPO. Include `system` in every
row for consistent prompting.

## GSM8K

Seed SFT: `problem`, `system`, `reasoning_solution`.
Pool: `problem`, `system`, `groud_truth_solution` (the historical spelling is
intentional). Reference solutions must expose a LaTeX boxed answer recognized
by `math-verify`.

```json
[{"problem":"Synthetic example: what is 1 + 1?","system":"Reason and give a boxed final answer.","groud_truth_solution":"\\boxed{2}"}]
```

Evaluation rows use `problem`, `system`, and `answer`. The merger parses `answer`
as a LaTeX reference, so expose a boxed answer such as `\\boxed{2}`. This differs
from the pool's `groud_truth_solution` key; convert the fields explicitly.

## Prepared DPO pairs

`instruction`, `system`, `chosen`, `rejected`; MedQA additionally retains `input`.
Both responses are strings. Optional preference seed files use this same format.
Ordinary SFT data is not a substitute for preference pairs. Dataset registration
for generated files is written automatically to `prepared_data/dataset_info.json`.

## H-GRPO

Local JSON input uses `instruction`, optional `input`, and `output` as the
reference solution. For GSM8K, put the math question in `instruction` and a
math-verifiable boxed reference in `output`. The adapter constructs `prompt`
and `solution`; the task-specific recipe supplies the system prompt.
