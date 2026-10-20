# HSIR 代码说明

本目录是论文 [Better, Faster: Harnessing Self-Improvement in Large Reasoning Models](https://arxiv.org/abs/2605.24998) 的独立代码整理版。
完整安装、训练与评测命令见 [英文 README](README.md)。

- `run.py`：可配置路径、GPU 数量和训练模式的入口；加 `--dry-run` 只预览命令。
- `src/`：论文方法的采样、评分、数据筛选和评测实现。
- `data/`：MedQA、GSM8K 的种子训练集、训练池、H-GRPO 输入和测试集。
- `docs/DATA.md`：数据文件、条数和字段说明。
- `H-GRPO/`：注意力加权 InDiv 奖励、分组归一化与训练配置。
- `docs/REPRODUCIBILITY.md`：论文配置、修复内容与尚未验证的部分。
- `tests/`：无需 GPU 的回归测试。
- `LICENSE`：Apache-2.0；第三方版权声明保留。

基础数据已包含在代码包内，文件条数和校验值见 `data/manifest.json`。模型权重和历史训练输出单独保存。
已验证代码语法、CPU 数值计算、数据处理及运行流程预览，没有运行 GPU 训练或完整复现实验。
训练、采样环境需要按 CUDA 与框架版本分别安装；依赖范围不是已验证的版本锁定文件。
