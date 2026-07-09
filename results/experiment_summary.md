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
| Transformer-CTC + ConvSubsampling + Fbank | 30 | 8.96% | 7.22% | stronger greedy baseline |

Conclusion:
The optimized 30-epoch ConvSubsampling + Fbank model achieved the best greedy result. SpecAugment still needs to be re-tested on top of this stronger baseline for a fair robustness comparison.
