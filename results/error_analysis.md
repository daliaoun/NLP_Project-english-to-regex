# Error analysis

## LSTM baseline, 15 epochs

### New regexes (test): 1884 examples, 1741 exact, 17 equivalent but written differently, 126 wrong (2 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 1031 | 88.1 | 89.5 |
| OR (|) | 597 | 96.6 | 97.0 |
| counter {n,} | 1037 | 90.5 | 91.9 |
| negation (~) | 1113 | 93.1 | 93.6 |
| two or more literals | 392 | 89.3 | 90.3 |

Wrong outputs by type: other structure error 91, wrong character class 19, wrong number only 10, negation missing or extra 4, invalid regex 2

### New phrasings: 1500 examples, 1126 exact, 33 equivalent but written differently, 341 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 923 | 70.5 | 73.1 |
| OR (|) | 530 | 80.8 | 82.5 |
| counter {n,} | 932 | 74.1 | 76.3 |
| negation (~) | 774 | 80.6 | 81.8 |
| two or more literals | 316 | 70.6 | 75.3 |

Wrong outputs by type: other structure error 225, negation missing or extra 75, wrong character class 30, wrong number only 11

### Handwritten: 50 examples, 47 exact, 1 equivalent but written differently, 2 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 8 | 87.5 | 100.0 |
| OR (|) | 4 | 75.0 | 75.0 |
| counter {n,} | 14 | 100.0 | 100.0 |
| negation (~) | 12 | 91.7 | 91.7 |
| two or more literals | 8 | 100.0 | 100.0 |

Wrong outputs by type: negation missing or extra 1, other structure error 1

## LSTM baseline, 30 epochs

### New regexes (test): 1884 examples, 1801 exact, 12 equivalent but written differently, 71 wrong (3 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 1031 | 92.5 | 93.6 |
| OR (|) | 597 | 99.0 | 99.2 |
| counter {n,} | 1037 | 94.1 | 95.3 |
| negation (~) | 1113 | 96.0 | 96.5 |
| two or more literals | 392 | 93.9 | 94.6 |

Wrong outputs by type: other structure error 49, wrong character class 11, wrong number only 5, invalid regex 3, negation missing or extra 3

### New phrasings: 1500 examples, 1218 exact, 32 equivalent but written differently, 250 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 923 | 76.6 | 79.6 |
| OR (|) | 530 | 87.5 | 88.3 |
| counter {n,} | 932 | 80.8 | 82.9 |
| negation (~) | 774 | 84.2 | 85.9 |
| two or more literals | 316 | 74.7 | 80.7 |

Wrong outputs by type: other structure error 168, negation missing or extra 48, wrong character class 19, wrong number only 15

### Handwritten: 50 examples, 46 exact, 0 equivalent but written differently, 4 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 8 | 75.0 | 75.0 |
| OR (|) | 4 | 100.0 | 100.0 |
| counter {n,} | 14 | 92.9 | 92.9 |
| negation (~) | 12 | 83.3 | 83.3 |
| two or more literals | 8 | 100.0 | 100.0 |

Wrong outputs by type: negation missing or extra 2, wrong connective (and/or) 2

## LSTM + attention

### New regexes (test): 1884 examples, 1872 exact, 1 equivalent but written differently, 11 wrong (2 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 1031 | 99.4 | 99.5 |
| OR (|) | 597 | 99.0 | 99.0 |
| counter {n,} | 1037 | 99.7 | 99.8 |
| negation (~) | 1113 | 99.6 | 99.6 |
| two or more literals | 392 | 98.5 | 98.5 |

Wrong outputs by type: other structure error 9, invalid regex 2

### New phrasings: 1500 examples, 1239 exact, 26 equivalent but written differently, 235 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 923 | 83.7 | 85.7 |
| OR (|) | 530 | 79.1 | 80.6 |
| counter {n,} | 932 | 83.7 | 85.1 |
| negation (~) | 774 | 88.4 | 89.3 |
| two or more literals | 316 | 79.1 | 82.9 |

Wrong outputs by type: other structure error 156, negation missing or extra 68, wrong character class 8, wrong number only 3

### Handwritten: 50 examples, 48 exact, 0 equivalent but written differently, 2 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 8 | 87.5 | 87.5 |
| OR (|) | 4 | 100.0 | 100.0 |
| counter {n,} | 14 | 92.9 | 92.9 |
| negation (~) | 12 | 91.7 | 91.7 |
| two or more literals | 8 | 100.0 | 100.0 |

Wrong outputs by type: negation missing or extra 1, wrong connective (and/or) 1

## LSTM + attention, seed 2

### New regexes (test): 1884 examples, 1879 exact, 0 equivalent but written differently, 5 wrong (2 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 1031 | 99.7 | 99.7 |
| OR (|) | 597 | 99.7 | 99.7 |
| counter {n,} | 1037 | 99.8 | 99.8 |
| negation (~) | 1113 | 99.9 | 99.9 |
| two or more literals | 392 | 99.7 | 99.7 |

Wrong outputs by type: other structure error 3, invalid regex 2

### New phrasings: 1500 examples, 1250 exact, 20 equivalent but written differently, 230 wrong (6 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 923 | 83.9 | 85.4 |
| OR (|) | 530 | 80.9 | 82.1 |
| counter {n,} | 932 | 83.6 | 84.4 |
| negation (~) | 774 | 89.0 | 89.8 |
| two or more literals | 316 | 83.9 | 86.1 |

Wrong outputs by type: other structure error 145, negation missing or extra 67, wrong character class 7, invalid regex 6, wrong number only 4, wrong connective (and/or) 1

### Handwritten: 50 examples, 48 exact, 0 equivalent but written differently, 2 wrong (0 invalid)

| Feature of the reference | n | Exact (%) | Same behaviour (%) |
|---|---|---|---|
| AND (&) | 8 | 87.5 | 87.5 |
| OR (|) | 4 | 100.0 | 100.0 |
| counter {n,} | 14 | 92.9 | 92.9 |
| negation (~) | 12 | 91.7 | 91.7 |
| two or more literals | 8 | 100.0 | 100.0 |

Wrong outputs by type: negation missing or extra 1, wrong connective (and/or) 1
