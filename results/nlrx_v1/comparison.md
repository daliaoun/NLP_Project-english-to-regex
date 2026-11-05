| Model | Params | Seeds | Test loss | Perplexity | Token acc (%) | BLEU-4 | Exact match (%) | Functional acc (%) | Functional, non-degenerate refs (%) | Valid regex (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_attn_aug | 1,408,683 | 1 | 0.134 | 1.144 | 96.0 | 85.4 | 45.3 | 65.1 | 60.8 | 99.9 |
| lstm_attn | 1,408,683 | 2 | 0.140 ± 0.006 | 1.151 ± 0.007 | 95.7 ± 0.1 | 84.8 ± 0.1 | 41.2 ± 0.9 | 62.4 ± 0.8 | 58.4 ± 1.0 | 99.8 ± 0.2 |
| lstm_attn_tf05 | 1,408,683 | 1 | 0.189 | 1.208 | 94.7 | 85.1 | 40.8 | 61.1 | 58.3 | 93.3 |
| lstm_base_long | 998,443 | 1 | 0.199 | 1.220 | 95.0 | 82.3 | 34.1 | 56.4 | 50.7 | 99.8 |
| lstm_base | 998,443 | 2 | 0.185 ± 0.000 | 1.203 ± 0.000 | 94.9 ± 0.1 | 81.7 ± 0.0 | 32.2 ± 1.2 | 55.0 ± 0.8 | 49.7 ± 0.5 | 99.9 ± 0.1 |
| lstm_biattn | 2,273,707 | 1 | 0.146 | 1.157 | 95.5 | 83.9 | 40.8 | 63.6 | 59.1 | 99.5 |

Non-degenerate = the reference regex accepts some but not all of the probe strings (otherwise 'both accept nothing' counts as equivalent).

Breakdown on the test set (exact match / functional accuracy, %):

| Model | EM, regex unseen in training | Functional, unseen | EM, regex seen in training | Best epoch | Epochs run |
|---|---|---|---|---|---|
| lstm_attn_aug | 45.2 | 65.1 | 47.3 | 20 | 20 |
| lstm_attn | 41.4 ± 0.8 | 62.6 ± 0.7 | 39.1 ± 2.7 | 19 ± 2 | 27 ± 2 |
| lstm_attn_tf05 | 40.8 | 61.1 | 41.8 | 25 | 30 |
| lstm_base_long | 34.1 | 56.8 | 34.5 | 41 | 53 |
| lstm_base | 31.7 ± 1.3 | 54.8 ± 0.8 | 40.9 ± 0.9 | 30 ± 0 | 30 ± 0 |
| lstm_biattn | 40.9 | 63.8 | 40.0 | 14 | 22 |

Decoding strategy (same trained model, test set):

| Run | Decoding | BLEU-4 | Exact match (%) | Functional acc (%) | Valid regex (%) |
|---|---|---|---|---|---|
| lstm_attn_aug_s1 | greedy | 85.4 | 45.3 | 65.1 | 99.9 |
| lstm_attn_aug_s1 | beam5 | 84.7 | 43.9 | 64.1 | 100.0 |
| lstm_attn_s1 | greedy | 84.9 | 40.3 | 61.7 | 99.7 |
| lstm_attn_s1 | beam5 | 84.1 | 39.6 | 61.3 | 100.0 |
| lstm_base_s1 | greedy | 81.8 | 33.3 | 55.8 | 99.9 |
| lstm_base_s1 | beam5 | 81.7 | 34.3 | 56.5 | 100.0 |