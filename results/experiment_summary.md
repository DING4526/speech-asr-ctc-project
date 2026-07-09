# 实验结果摘要

项目：Transformer Encoder + CTC 中文拼音语音识别

数据集：
- 使用 Lab6 Biaobei ASR 划分，并复制到 `data/lab6_clean`。
- 清洗后 train/dev/test 数量为 7999/1000/1000。
- 移除了一个不可读音频：`006230.wav`。

主结果：

| 实验 | 最佳轮次 | Dev PER | Test PER | 说明 |
|---|---:|---:|---:|---|
| 原 final Transformer-CTC baseline | 30 | 13.16% | 15.69% | 队友已完成的原始主结果 |
| E1 ConvSubsampling + log-Mel | 45 | 11.00% | 11.07% | 卷积下采样前端消融，45 轮后基本平台化 |
| E2 ConvSubsampling + Fbank | 81 | 6.05% | 5.06% | 强 greedy baseline |
| E3 ConvSubsampling + Fbank + weak SpecAugment | 84 | 4.71% | 4.33% | 在强 baseline 上进行弱增强 |

鲁棒性评估：

| 条件 | E2 PER | E3 PER | E3 相对 E2 |
|---|---:|---:|---:|
| clean | 5.06% | 4.33% | -0.73 pp |
| noise20 | 25.96% | 22.49% | -3.47 pp |
| noise10 | 71.29% | 67.63% | -3.66 pp |
| noise5 | 90.43% | 88.05% | -2.38 pp |
| speed0.9 | 40.39% | 33.57% | -6.82 pp |
| speed1.1 | 40.56% | 34.55% | -6.01 pp |
| volume0.5 | 5.11% | 4.38% | -0.73 pp |
| volume1.5 | 5.08% | 4.33% | -0.75 pp |
| echo | 27.34% | 24.28% | -3.06 pp |

结论：
固定 30 epoch 会低估 SpecAugment，因为 E3 在 30 轮时尚未收敛。按 dev PER 继续续训到收敛后，E3 在 clean test 和全部扰动条件下都优于 E2，说明弱 SpecAugment 在当前强 baseline 上不仅没有损害 clean 性能，还带来了更好的鲁棒性。强噪声和速度扰动仍是主要错误来源，后续可考虑显式加入噪声/速度增强或使用学习率退火继续细化。
