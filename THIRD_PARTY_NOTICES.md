# Third-party software

This release uses external libraries including PyTorch, Transformers, vLLM,
Hugging Face Datasets and LLaMA-Factory. Install these independently under their
respective licenses. Model weights and benchmark content are not covered by this project's
code license. Bundled text datasets retain their source dataset terms; model
weights and images are supplied separately.

`H-GRPO/open_r1/` contains a modified snapshot of Hugging Face Open-R1
(https://github.com/huggingface/open-r1), and `H-GRPO/hsir_grpo_trainer.py`
contains a modified TRL trainer (https://github.com/huggingface/trl).
These files retain their Apache-2.0 copyright headers; the license text is in
`LICENSE`. The exact upstream commit was not present in the extracted snapshot.
The local trainer, reward integration, launchers and recipes are modified for
HSIR. The H-GRPO scoring and reward implementation was updated on 2026-10-20.
