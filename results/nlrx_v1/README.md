# Archived: first version on NL-RX-Turk

The first version of the project trained the same architecture on NL-RX-Turk (Locascio et al., 2016).
Best result: 45.3% exact match, 65.1% functional accuracy. It was replaced by the verified VeriRegex
dataset after we found that some NL-RX labels contradict their English (see results/REPORT.md, 3.1).
This folder keeps its report, tables, figures, logs, metrics and slide sources for reference; the
model checkpoints are not included in the packaged project. The runs used `src/train.py --dataset NL-RX-Turk`
with the NL-RX data cloned into data_raw/deep-regex (see the archived REPORT.md).
Note: the beam-search numbers in that report came from a beam search that stopped too early (fixed since).
