# Error analysis

## lstm_attn_aug_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 52.6 | 67.1 |
| contains alternation (|) | 368 | 48.1 | 60.9 |
| contains negation (~) | 154 | 42.2 | 55.2 |
| contains word boundary (\b) | 194 | 39.2 | 58.2 |
| long regex (>= 30 tokens) | 232 | 52.2 | 70.3 |
| contains intersection (&) | 271 | 45.8 | 71.6 |
| long description (>= 15 words) | 112 | 51.8 | 69.6 |

Of 999 test examples: 453 exact matches, 197 behaviourally equivalent but written differently, 348 valid but wrong, 1 invalid regexes.

Among the 348 valid-but-wrong outputs, 127 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_attn_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 43.7 | 62.0 |
| contains alternation (|) | 368 | 45.1 | 57.9 |
| contains negation (~) | 154 | 35.1 | 48.7 |
| contains word boundary (\b) | 194 | 36.6 | 57.7 |
| long regex (>= 30 tokens) | 232 | 47.0 | 68.5 |
| contains intersection (&) | 271 | 42.4 | 67.9 |
| long description (>= 15 words) | 112 | 48.2 | 67.0 |

Of 999 test examples: 403 exact matches, 213 behaviourally equivalent but written differently, 380 valid but wrong, 3 invalid regexes.

Among the 380 valid-but-wrong outputs, 160 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_attn_s2

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 47.9 | 69.0 |
| contains alternation (|) | 368 | 45.9 | 61.1 |
| contains negation (~) | 154 | 34.4 | 47.4 |
| contains word boundary (\b) | 194 | 38.7 | 57.2 |
| long regex (>= 30 tokens) | 232 | 45.7 | 65.5 |
| contains intersection (&) | 271 | 42.4 | 66.4 |
| long description (>= 15 words) | 112 | 51.8 | 70.5 |

Of 999 test examples: 421 exact matches, 210 behaviourally equivalent but written differently, 368 valid but wrong, 0 invalid regexes.

Among the 368 valid-but-wrong outputs, 164 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_attn_tf05_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 49.8 | 65.3 |
| contains alternation (|) | 368 | 44.8 | 58.4 |
| contains negation (~) | 154 | 31.8 | 47.4 |
| contains word boundary (\b) | 194 | 37.1 | 58.2 |
| long regex (>= 30 tokens) | 232 | 43.1 | 64.2 |
| contains intersection (&) | 271 | 41.3 | 65.7 |
| long description (>= 15 words) | 112 | 49.1 | 67.9 |

Of 999 test examples: 408 exact matches, 202 behaviourally equivalent but written differently, 322 valid but wrong, 67 invalid regexes.

Among the 322 valid-but-wrong outputs, 133 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_base_long_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 38.0 | 59.6 |
| contains alternation (|) | 368 | 34.5 | 51.4 |
| contains negation (~) | 154 | 26.6 | 42.2 |
| contains word boundary (\b) | 194 | 33.0 | 53.6 |
| long regex (>= 30 tokens) | 232 | 31.5 | 58.2 |
| contains intersection (&) | 271 | 38.0 | 66.4 |
| long description (>= 15 words) | 112 | 40.2 | 66.1 |

Of 999 test examples: 341 exact matches, 222 behaviourally equivalent but written differently, 434 valid but wrong, 2 invalid regexes.

Among the 434 valid-but-wrong outputs, 161 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_base_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 35.2 | 57.3 |
| contains alternation (|) | 368 | 32.6 | 50.5 |
| contains negation (~) | 154 | 26.0 | 40.9 |
| contains word boundary (\b) | 194 | 35.1 | 54.6 |
| long regex (>= 30 tokens) | 232 | 30.2 | 56.9 |
| contains intersection (&) | 271 | 35.1 | 64.6 |
| long description (>= 15 words) | 112 | 33.0 | 61.6 |

Of 999 test examples: 333 exact matches, 224 behaviourally equivalent but written differently, 441 valid but wrong, 1 invalid regexes.

Among the 441 valid-but-wrong outputs, 164 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_base_s2

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 26.8 | 46.9 |
| contains alternation (|) | 368 | 34.2 | 50.5 |
| contains negation (~) | 154 | 25.3 | 44.2 |
| contains word boundary (\b) | 194 | 32.5 | 52.1 |
| long regex (>= 30 tokens) | 232 | 31.9 | 54.7 |
| contains intersection (&) | 271 | 32.1 | 63.8 |
| long description (>= 15 words) | 112 | 30.4 | 59.8 |

Of 999 test examples: 310 exact matches, 232 behaviourally equivalent but written differently, 457 valid but wrong, 0 invalid regexes.

Among the 457 valid-but-wrong outputs, 164 are 'near misses' (character similarity >= 0.85 with the reference).

## lstm_biattn_s1

Accuracy by property of the reference regex / description (test set):

| Property | n | Exact match (%) | Functional (%) |
|---|---|---|---|
| contains repetition {n,m} | 213 | 44.1 | 63.8 |
| contains alternation (|) | 368 | 45.1 | 60.1 |
| contains negation (~) | 154 | 35.1 | 50.0 |
| contains word boundary (\b) | 194 | 38.1 | 59.8 |
| long regex (>= 30 tokens) | 232 | 44.0 | 67.2 |
| contains intersection (&) | 271 | 41.3 | 69.0 |
| long description (>= 15 words) | 112 | 45.5 | 68.8 |

Of 999 test examples: 408 exact matches, 227 behaviourally equivalent but written differently, 359 valid but wrong, 5 invalid regexes.

Among the 359 valid-but-wrong outputs, 143 are 'near misses' (character similarity >= 0.85 with the reference).
