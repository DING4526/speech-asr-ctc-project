# Transformer-CTC Chinese Pinyin ASR

This project is a final project for Speech Information Processing. It extends the Lab6 ASR task and implements a Chinese pinyin automatic speech recognition system based on Transformer Encoder and CTC loss.

## Task

Input: Mandarin speech waveform.

Output: pinyin token sequence without tone.

## Method

The system uses log-Mel or Kaldi-style Fbank features, Transformer Encoder acoustic modeling, CTC loss, greedy CTC decoding, and PER / WER-like token error rate for evaluation. The optimized experiment line also supports a 4x convolutional subsampling frontend before the Transformer encoder.

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
| Transformer-CTC + ConvSubsampling + log-Mel | 30 | 11.05% | 11.97% |
| Transformer-CTC + ConvSubsampling + Fbank | 30 | 8.96% | 7.22% |
| Transformer-CTC + ConvSubsampling + Fbank + weak SpecAugment | 30 | 9.11% | 7.83% |

The best current greedy result is obtained by the 30-epoch ConvSubsampling + Fbank model.
Weak SpecAugment did not improve the stronger baseline under the current settings, but it remained close to the optimized greedy baseline and far better than the earlier 15-epoch augmentation run.

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
| E0 final baseline | configs/e0_final_baseline.yaml | Re-evaluate the current final baseline with unified prediction and error-analysis outputs. |
| E1 conv log-Mel | configs/e1_conv_logmel.yaml | Isolate the effect of the 4x convolutional subsampling frontend. |
| E2 conv Fbank strong | configs/e2_conv_fbank_strong.yaml | Train the stronger greedy baseline using the Exp6-inspired frontend, Fbank, and optimizer recipe. |
| E3 conv Fbank + weak SpecAugment | configs/e3_conv_fbank_weak_specaugment.yaml | Test lightweight robustness augmentation on top of the stronger baseline. |

Recommended order:

1. Run `configs/smoke_conv_fbank.yaml` to verify the new frontend and Fbank backend.
2. Train E1 and E2, then evaluate both test sets with `--predictions-out` and `--analysis-out`.
3. Train E3 only after E2 is stable, and compare the E2/E3 curves and error breakdowns.
4. Generate summary figures from all finished experiment outputs.

Summarize finished experiments:

python scripts/summarize_experiments.py

Continue the strong baseline and SpecAugment runs to 45 epochs:

python train.py --config configs/e1_conv_logmel_45.yaml --resume checkpoints/e1_conv_logmel/last.pt

python train.py --config configs/e2_conv_fbank_strong_45.yaml --resume checkpoints/e2_conv_fbank_strong/last.pt

python train.py --config configs/e3_conv_fbank_weak_specaugment_45.yaml --resume checkpoints/e3_conv_fbank_weak_specaugment/last.pt

If the dev PER is still improving at 45 epochs, continue to 60 epochs:

python train.py --config configs/e2_conv_fbank_strong_60.yaml --resume checkpoints/e2_conv_fbank_strong/last.pt

python train.py --config configs/e3_conv_fbank_weak_specaugment_60.yaml --resume checkpoints/e3_conv_fbank_weak_specaugment/last.pt

If the best dev PER is still refreshed near epoch 60, continue to 75 epochs:

python train.py --config configs/e2_conv_fbank_strong_75.yaml --resume checkpoints/e2_conv_fbank_strong/last.pt

python train.py --config configs/e3_conv_fbank_weak_specaugment_75.yaml --resume checkpoints/e3_conv_fbank_weak_specaugment/last.pt

Evaluate waveform-level robustness:

python scripts/evaluate_robustness.py --config configs/e2_conv_fbank_strong_45.yaml --checkpoint checkpoints/e2_conv_fbank_strong/best.pt --summary-out results/e2_45_robustness_summary.csv --records-out results/e2_45_robustness_records.csv --analysis-out results/e2_45_robustness_analysis.json

python scripts/evaluate_robustness.py --config configs/e3_conv_fbank_weak_specaugment_45.yaml --checkpoint checkpoints/e3_conv_fbank_weak_specaugment/best.pt --summary-out results/e3_45_robustness_summary.csv --records-out results/e3_45_robustness_records.csv --analysis-out results/e3_45_robustness_analysis.json

Plot robustness comparison:

python scripts/plot_robustness.py

## Members

- Li Zehao 2312594
- Ding Yisan 2310724
