# Experiment Log

## Project scope
Character-Level Language Modeling with Char-RNN: Effects of Dataset and Model Capacity on Generated Text Quality

## Fixed experiment matrix
- Dataset: Tiny Shakespeare, Sherlock Holmes
- Model type: GRU, LSTM
- Hidden size: 128, 256
- Temperatures: 0.5, 0.8, 1.0

## Run table
| Run name | Dataset | Model | Hidden size | Epochs | Final loss | Notes |
|---|---|---|---:|---:|---:|---|
| shakespeare_gru_h128 | Shakespeare | GRU | 128 | 1200 | 1.3571 | Stable dialogue-like structure; more repetition than h256 |
| shakespeare_gru_h256 | Shakespeare | GRU | 256 | 1200 | 1.2518 | Best Shakespeare run by loss; more coherent role-style outputs |
| shakespeare_lstm_h128 | Shakespeare | LSTM | 128 | 1200 | 1.4616 | Weakest Shakespeare run by loss; more noisy text |
| shakespeare_lstm_h256 | Shakespeare | LSTM | 256 | 1200 | 1.3154 | Better than LSTM-128 but behind GRU runs |
| sherlock_gru_h128 | Sherlock | GRU | 128 | 1200 | 1.2629 | Good sentence flow; occasional malformed words |
| sherlock_gru_h256 | Sherlock | GRU | 256 | 1200 | 1.1274 | Best overall run by loss; strongest local coherence |
| sherlock_lstm_h128 | Sherlock | LSTM | 128 | 1200 | 1.4069 | Coherence improves over epochs but still noisy |
| sherlock_lstm_h256 | Sherlock | LSTM | 256 | 1200 | 1.2308 | Stronger than LSTM-128; still below GRU-256 |

## Qualitative sample notes
- Compare coherence and syntax by temperature.
- Track failure modes: repetition loops, gibberish, punctuation collapse.
- Note style transfer quality by dataset.
