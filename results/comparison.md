## Main results (greedy decoding)

| Model | Params | Test set | n | Exact match (%) | Same behaviour (%) | BLEU-4 | Valid regex (%) | Perplexity |
|---|---|---|---|---|---|---|---|---|
| LSTM baseline, 15 epochs | 964,396 | New regexes (test) | 1884 | 92.4 | 93.3 | 98.0 | 99.9 | 1.017 |
| LSTM baseline, 15 epochs | 964,396 | New phrasings | 1500 | 75.1 | 77.3 | 95.6 | 100.0 | 1.053 |
| LSTM baseline, 15 epochs | 964,396 | Handwritten | 50 | 94.0 | 96.0 | 98.2 | 100.0 | 1.033 |
| LSTM baseline, 30 epochs | 964,396 | New regexes (test) | 1884 | 95.6 | 96.2 | 99.0 | 99.8 | 1.009 |
| LSTM baseline, 30 epochs | 964,396 | New phrasings | 1500 | 81.2 | 83.3 | 96.8 | 100.0 | 1.035 |
| LSTM baseline, 30 epochs | 964,396 | Handwritten | 50 | 92.0 | 92.0 | 97.4 | 100.0 | 1.067 |
| LSTM + attention | 1,375,020 | New regexes (test) | 1884 | 99.4 | 99.4 | 99.9 | 99.9 | 1.002 |
| LSTM + attention | 1,375,020 | New phrasings | 1500 | 82.6 | 84.3 | 97.4 | 100.0 | 1.039 |
| LSTM + attention | 1,375,020 | Handwritten | 50 | 96.0 | 96.0 | 97.6 | 100.0 | 1.038 |
| LSTM + attention, seed 2 | 1,375,020 | New regexes (test) | 1884 | 99.7 | 99.7 | 100.0 | 99.9 | 1.001 |
| LSTM + attention, seed 2 | 1,375,020 | New phrasings | 1500 | 83.3 | 84.7 | 97.0 | 99.6 | 1.036 |
| LSTM + attention, seed 2 | 1,375,020 | Handwritten | 50 | 96.0 | 96.0 | 97.6 | 100.0 | 1.039 |

## By kind of description (test set, exact match / same behaviour, %)

| Model | One condition | Negated condition | Two or three conditions with AND | Two conditions with OR |
|---|---|---|---|---|
| LSTM baseline, 15 epochs | 100.0 / 100.0 (n=20) | 100.0 / 100.0 (n=236) | 88.1 / 89.5 (n=1031) | 96.6 / 97.0 (n=597) |
| LSTM baseline, 30 epochs | 100.0 / 100.0 (n=20) | 100.0 / 100.0 (n=236) | 92.5 / 93.6 (n=1031) | 99.0 / 99.2 (n=597) |
| LSTM + attention | 100.0 / 100.0 (n=20) | 100.0 / 100.0 (n=236) | 99.4 / 99.5 (n=1031) | 99.0 / 99.0 (n=597) |
| LSTM + attention, seed 2 | 100.0 / 100.0 (n=20) | 100.0 / 100.0 (n=236) | 99.7 / 99.7 (n=1031) | 99.7 / 99.7 (n=597) |

## Greedy versus beam search (same trained model)

| Model | Test set | Decoding | Exact match (%) | Same behaviour (%) | Valid regex (%) |
|---|---|---|---|---|---|
| LSTM baseline, 15 epochs | New regexes (test) | greedy | 92.4 | 93.3 | 99.9 |
| LSTM baseline, 15 epochs | New regexes (test) | beam5 | 92.7 | 93.5 | 100.0 |
| LSTM baseline, 15 epochs | New phrasings | greedy | 75.1 | 77.3 | 100.0 |
| LSTM baseline, 15 epochs | New phrasings | beam5 | 75.1 | 77.5 | 100.0 |
| LSTM baseline, 15 epochs | Handwritten | greedy | 94.0 | 96.0 | 100.0 |
| LSTM baseline, 15 epochs | Handwritten | beam5 | 96.0 | 96.0 | 100.0 |
| LSTM + attention | New regexes (test) | greedy | 99.4 | 99.4 | 99.9 |
| LSTM + attention | New regexes (test) | beam5 | 99.4 | 99.5 | 99.9 |
| LSTM + attention | New phrasings | greedy | 82.6 | 84.3 | 100.0 |
| LSTM + attention | New phrasings | beam5 | 82.6 | 84.3 | 99.9 |
| LSTM + attention | Handwritten | greedy | 96.0 | 96.0 | 100.0 |
| LSTM + attention | Handwritten | beam5 | 96.0 | 96.0 | 100.0 |

## Training

| Model | Best epoch | Epochs run | Minutes | Best validation exact match (%) |
|---|---|---|---|---|
| LSTM baseline, 15 epochs | 14 | 15 | 19 | 94.3 |
| LSTM baseline, 30 epochs | 28 | 30 | 51 | 96.3 |
| LSTM + attention | 7 | 11 | 25 | 100.0 |
| LSTM + attention, seed 2 | 7 | 11 | 39 | 99.9 |