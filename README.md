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

If epoch 75 is still at or near the best dev PER, continue to 90 epochs:

python train.py --config configs/e2_conv_fbank_strong_90.yaml --resume checkpoints/e2_conv_fbank_strong/last.pt

python train.py --config configs/e3_conv_fbank_weak_specaugment_90.yaml --resume checkpoints/e3_conv_fbank_weak_specaugment/last.pt

Evaluate waveform-level robustness:

python scripts/evaluate_robustness.py --config configs/e2_conv_fbank_strong_90.yaml --checkpoint checkpoints/e2_conv_fbank_strong/best.pt --summary-out results/e2_90_robustness_summary.csv --records-out results/e2_90_robustness_records.csv --analysis-out results/e2_90_robustness_analysis.json

python scripts/evaluate_robustness.py --config configs/e3_conv_fbank_weak_specaugment_90.yaml --checkpoint checkpoints/e3_conv_fbank_weak_specaugment/best.pt --summary-out results/e3_90_robustness_summary.csv --records-out results/e3_90_robustness_records.csv --analysis-out results/e3_90_robustness_analysis.json

Plot robustness comparison:

python scripts/plot_robustness.py

Next-stage robustness and decoder analysis:

python train.py --config configs/r1_e3_audio_mild_99.yaml --resume checkpoints/e3_conv_fbank_weak_specaugment/best.pt --reset-optimizer

python train.py --config configs/r2_e3_audio_mixed_125.yaml --resume checkpoints/r2_e3_audio_mixed/last.pt

python scripts/select_robust_checkpoint.py --config configs/r2_e3_audio_mixed_125.yaml --checkpoint-dir checkpoints/r2_e3_audio_mixed --checkpoints checkpoints/r2_e3_audio_mixed/epoch_119.pt,checkpoints/r2_e3_audio_mixed/epoch_125.pt --conditions clean,noise20,speed0.9,speed1.1,echo --summary-out results/r2_125_robust_dev_summary.csv --analysis-out results/r2_125_robust_dev_analysis.json --selected-out checkpoints/r2_e3_audio_mixed/robust_best.pt

python scripts/evaluate_decoder.py --config configs/r2_e3_audio_mixed_125.yaml --checkpoint checkpoints/r2_e3_audio_mixed/robust_best.pt --split test --decoder prefix_beam_lm --beam-width 25 --topk 12 --lm-alpha 0.3 --lm-beta 0.0 --experiment D3_R2_beam_lm --matrix-out results/decoder_matrix.csv

python scripts/build_final_matrices.py --robust-input R0_E3_specaugment:results/e3_90_robustness_summary.csv --robust-input R1_mild_audio_aug:results/r1_robustness_summary.csv --robust-input R2_mixed_audio_aug:results/r2_robustness_summary.csv

python scripts/plot_optimization_summary.py --decoder-matrix results/decoder_matrix.csv --robust-matrix results/final_robustness_matrix.csv --substitution E3_greedy:results/d0_e3_greedy_substitutions.csv --substitution E3_LM:results/d2_e3_beam_lm_substitutions.csv --substitution R2_greedy:results/d3_r2_greedy_substitutions.csv --substitution R2_LM:results/d3_r2_beam_lm_substitutions.csv

## Members

- Li Zehao 2312594
- Ding Yisan 2310724
