# Char-RNN Final Project

Project scope (locked):
`Character-Level Language Modeling with Char-RNN: Effects of Dataset and Model Capacity on Generated Text Quality`

This repository is based on the PyTorch approach from `spro/char-rnn.pytorch`, updated for modern PyTorch and extended with reproducible experiment tooling.

## Folder layout

- `data/`
  - `shakespeare.txt`
  - `sherlock.txt`
- `results/` (checkpoints, loss logs, run metadata)
- `samples/` (generated text)
- `report_notes/` (analysis notes for final writeup)

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/download_datasets.py
```

## First milestone (Shakespeare pipeline check)

Train one run and save losses + sample at temperature `0.8`:

```bash
python train.py data/shakespeare.txt ^
  --run_name shakespeare_gru_h128 ^
  --model gru ^
  --hidden_size 128 ^
  --n_layers 2 ^
  --n_epochs 300 ^
  --print_every 100 ^
  --batch_size 64 ^
  --chunk_len 150 ^
  --learning_rate 0.003 ^
  --sample_temperature 0.8
```

This produces:
- checkpoint: `results/shakespeare_gru_h128.pt`
- loss log: `results/shakespeare_gru_h128_loss.csv`
- generated sample: `samples/shakespeare_gru_h128_temp0.8.txt`

## Generate text from a trained checkpoint

```bash
python generate.py results/shakespeare_gru_h128.pt ^
  --prime_str "Wh" ^
  --predict_len 500 ^
  --temperature 0.8
```

## Full experiment matrix

Runs all combinations:
- dataset: Shakespeare, Sherlock
- hidden size: 128, 256
- model: GRU, LSTM
- generation temperature: 0.5, 0.8, 1.0

```bash
python run_experiments.py
```
