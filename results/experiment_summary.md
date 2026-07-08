# Experiment Summary

Project: Transformer Encoder + CTC Chinese Pinyin ASR

Dataset:
- Lab6 Biaobei ASR split
- Cleaned train/dev/test manifests
- One unreadable audio file was removed: 006230.wav

Main Results:

| Experiment | Epochs | Dev PER | Test PER | Notes |
|---|---:|---:|---:|---|
| Transformer-CTC Baseline | 15 | 23.64% | see baseline15_test_result.txt | baseline model |
| Transformer-CTC + SpecAugment | 15 | 26.77% | 27.59% | augmentation did not improve performance |
| Transformer-CTC Baseline | 30 | 13.16% | 15.69% | final main result |

Conclusion:
The 30-epoch Transformer-CTC baseline achieved the best result. SpecAugment did not improve performance under the current lightweight CPU training setting.
