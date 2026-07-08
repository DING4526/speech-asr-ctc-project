# Transformer-CTC Chinese Pinyin ASR

This project is a final project for Speech Information Processing. It extends the Lab6 ASR task and implements a Chinese pinyin automatic speech recognition system based on Transformer Encoder and CTC loss.

## Task

Input: Mandarin speech waveform.

Output: pinyin token sequence without tone.

## Method

The system uses log-Mel filterbank features, Transformer Encoder acoustic modeling, CTC loss, greedy CTC decoding, and PER / WER-like token error rate for evaluation.

## Dataset

The experiment uses the Lab6 Biaobei ASR dataset split.

During preprocessing, one unreadable audio file, 006230.wav, was removed from the training manifest.

Cleaned data split:

- train: 7999 utterances
- dev: 1000 utterances
- test: 1000 utterances

The dataset itself is not included in this repository.

## Main Results

| Experiment | Epochs | Dev PER | Test PER |
|---|---:|---:|---:|
| Transformer-CTC Baseline | 15 | 23.64% | see results/baseline15_test_result.txt |
| Transformer-CTC + SpecAugment | 15 | 26.77% | 27.59% |
| Transformer-CTC Baseline | 30 | 13.16% | 15.69% |

The best result is obtained by the 30-epoch Transformer-CTC baseline model.

## Run

Train:

python train.py --config configs/lab6_clean_baseline_30.yaml

Evaluate:

python evaluate.py --config configs/lab6_clean_baseline_30.yaml --checkpoint checkpoints/lab6_clean_baseline_30/best.pt --split test

## Members

- Li Zehao 2312594
- Ding Yisan 2310724
