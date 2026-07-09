# Transformer-CTC Chinese Pinyin ASR

This project is a final project for Speech Information Processing. It extends the Lab6 ASR task and implements a Chinese pinyin automatic speech recognition system based on Transformer Encoder and CTC loss.

## Task

Input: Mandarin speech waveform.

Output: pinyin token sequence without tone.

## Method

The system uses log-Mel filterbank features, Transformer Encoder acoustic modeling, CTC loss, greedy CTC decoding, and PER / WER-like token error rate for evaluation.

## Dataset

The experiment uses the Lab6 Biaobei ASR dataset split from Exp6.

In this local workspace, the raw Exp6 dataset has been copied from:

`D:\03_Courses\3_2_06_Voice_information_processing\exp6\dataset`

to:

`data/lab6`

The cleaned manifests used by the current configs are under:

`data/lab6_clean`

During preprocessing, one unreadable audio file, 006230.wav, was removed from the training manifest.

Cleaned data split:

- train: 7999 utterances
- dev: 1000 utterances
- test: 1000 utterances

The dataset itself is not included in this repository.

Prepare the copied Exp6 data:

python scripts/prepare_exp6_data.py --raw-root data/lab6 --out-root data/lab6_clean

## Environment

The project is intended to run in the same environment as Exp6:

conda activate melotts

In this local machine, the environment Python is:

`C:\Users\ding1\anaconda3\envs\melotts\python.exe`

Verified local packages include `torch 2.7.1+cu118` and `torchaudio 2.7.1+cu118`; CUDA is available on an NVIDIA GeForce RTX 4060 Laptop GPU.

## Main Results

| Experiment | Epochs | Dev PER | Test PER |
|---|---:|---:|---:|
| Transformer-CTC Baseline | 15 | 23.64% | see results/baseline15_test_result.txt |
| Transformer-CTC + SpecAugment | 15 | 26.77% | 27.59% |
| Transformer-CTC Baseline | 30 | 13.16% | 15.69% |

The best result is obtained by the 30-epoch Transformer-CTC baseline model.
The 15-epoch SpecAugment result should be treated as an initial negative result rather than a final conclusion, because augmentation can slow convergence and needs a matched-epoch comparison.

## Run

Train:

python train.py --config configs/lab6_clean_baseline_30.yaml

Training writes checkpoints and an epoch-level `metrics.csv` under the configured `output_dir`.

Resume an interrupted run:

python train.py --config configs/lab6_clean_baseline_30.yaml --resume checkpoints/lab6_clean_baseline_30/last.pt

Evaluate:

python evaluate.py --config configs/lab6_clean_baseline_30.yaml --checkpoint checkpoints/lab6_clean_baseline_30/best.pt --split test

Export predictions and error analysis:

python evaluate.py --config configs/lab6_clean_baseline_30.yaml --checkpoint checkpoints/lab6_clean_baseline_30/best.pt --split test --predictions-out results/baseline30_predictions.csv --analysis-out results/baseline30_analysis.json

The analysis JSON includes total PER, substitution/deletion/insertion counts, PER by reference length, and worst prediction examples.

## Recommended Next Experiments

| Experiment | Config | Purpose |
|---|---|---|
| Matched SpecAugment | configs/lab6_clean_specaugment_30.yaml | Compare SpecAugment with the 30-epoch baseline under the same epoch budget. |
| Weak SpecAugment | configs/lab6_clean_specaugment_30_weak.yaml | Test whether milder masks reduce the underfitting observed in the 15-epoch run. |

Recommended order:

1. Run `configs/lab6_clean_specaugment_30_weak.yaml`.
2. Evaluate dev and test with `--analysis-out`.
3. Compare `metrics.csv` curves and deletion/insertion/substitution counts with the 30-epoch baseline.

## Members

- Li Zehao 2312594
- Ding Yisan 2310724
