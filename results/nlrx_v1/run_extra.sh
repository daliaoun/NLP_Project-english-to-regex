#!/bin/bash
# Extra experiments: second seed (variance), bidirectional encoder + attention, teacher-forcing 0.5
cd "$(dirname "$0")"
(python3 -I src/train.py --name lstm_attn_s2 --attention --epochs 30 --seed 2 > logs/lstm_attn_s2.log 2>&1 && \
 python3 -I src/train.py --name lstm_attn_tf05_s1 --attention --tf 0.5 --epochs 30 --seed 1 > logs/lstm_attn_tf05_s1.log 2>&1) &
(python3 -I src/train.py --name lstm_base_s2 --epochs 30 --seed 2 > logs/lstm_base_s2.log 2>&1 && \
 python3 -I src/train.py --name lstm_biattn_s1 --attention --bidirectional --epochs 30 --seed 1 > logs/lstm_biattn_s1.log 2>&1) &
wait
