# 实验结果摘要

项目：Transformer Encoder + CTC 中文拼音语音识别

数据集：
- 使用 Lab6 Biaobei ASR 划分，并复制到 `data/lab6_clean`。
- 清洗后 train/dev/test 数量为 7999/1000/1000。
- 移除了一个不可读音频：`006230.wav`。

## 主线一：收敛后的强 baseline

| 实验 | 最佳轮次 | Dev PER | Test PER | 说明 |
|---|---:|---:|---:|---|
| 原 final Transformer-CTC baseline | 30 | 13.16% | 15.69% | 队友已完成的原始主结果 |
| E1 ConvSubsampling + log-Mel | 45 | 11.00% | 11.07% | 卷积下采样前端消融，45 轮后基本平台化 |
| E2 ConvSubsampling + Fbank | 81 | 6.05% | 5.06% | 强 greedy baseline |
| E3 ConvSubsampling + Fbank + weak SpecAugment | 84 | 4.71% | 4.33% | 收敛后优于 E2，30 轮结论存在欠收敛偏差 |

## 主线二：波形级鲁棒性增强

| 条件 | R0 E3 | R1 mild audio aug | R2 mixed audio aug |
|---|---:|---:|---:|
| clean | 4.33% | 3.18% | 2.86% |
| noise20 | 22.49% | 5.40% | 4.24% |
| noise10 | 67.63% | 14.04% | 8.85% |
| noise5 | 88.05% | 36.35% | 19.23% |
| speed0.9 | 33.57% | 17.47% | 7.07% |
| speed1.1 | 34.55% | 18.75% | 7.90% |
| volume0.5 | 4.38% | 3.13% | 2.82% |
| volume1.5 | 4.33% | 3.20% | 2.87% |
| echo | 24.28% | 9.04% | 6.40% |

R2 使用 noise/speed/volume/echo 混合波形增强，从 E3 best checkpoint 微调到 125 轮，并用 dev 集的 mean(clean, noise20, speed0.9, speed1.1, echo) 选择 robust checkpoint。最终 R2 在 clean 和所有扰动条件下都优于 R0，没有出现 clean 场景 trade-off。

## 辅助线：解码与 substitution

| 实验 | Decoder | Test PER | S | D | I | Decode time |
|---|---|---:|---:|---:|---:|---:|
| D0 E3 greedy | greedy | 4.33% | 621 | 37 | 108 | 0.03s |
| D1 E3 beam | prefix beam | 4.26% | 625 | 32 | 96 | 19.63s |
| D2 E3 beam + 4-gram LM | prefix beam + LM | 3.73% | 566 | 65 | 28 | 82.14s |
| D3 R2 greedy | greedy | 2.86% | 437 | 20 | 49 | 0.04s |
| D3 R2 beam + 4-gram LM | prefix beam + LM | 2.32% | 367 | 36 | 7 | 83.89s |

Beam 本身只带来很小收益，且没有降低 substitution；拼音 4-gram LM 能明显降低 substitution，但推理成本很高。因此解码优化适合作为错误类型分析和上限参考，不作为主优化路线。主要贡献应放在波形级鲁棒性增强。

## 可视化产物

- `results/figures/final_robustness_per.png`：各扰动条件 PER 对比。
- `results/figures/clean_vs_robust_tradeoff.png`：clean PER 与平均扰动 PER trade-off。
- `results/figures/decoder_error_breakdown.png`：decoder 的 S/D/I 和 PER 对比。
- `results/figures/substitution_confusions.png`：高频 substitution 混淆对比。

结论：
本阶段最有价值的优化不是继续堆模型参数或延长普通训练，而是显式加入声学扰动增强。R2 mixed audio augmentation 将 clean test PER 从 E3 的 4.33% 降到 2.86%，同时把 noise20 从 22.49% 降到 4.24%、speed0.9 从 33.57% 降到 7.07%、echo 从 24.28% 降到 6.40%。LM 解码进一步把 R2 clean test PER 降到 2.32%，但推理耗时约 84 秒，只作为 substitution 辅助优化和分析参考。
