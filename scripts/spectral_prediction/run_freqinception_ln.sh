#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --split_data_path ./dataset/split_data \
  --continuum_filename continuum.csv \
  --normalized_filename normalized.csv \
  --labels_filename labels.csv \
  --model_id FreqInceptionLNet_spectral \
  --model FreqInceptionLNet \
  --data spectral \
  --features M \
  --feature_size 4700 \
  --label_size 4 \
  --stats_path conf/stats.yaml \
  --model_conf conf/freqinception_ln.yaml \
  --patience 20000 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --loss 'mse' \
  --lradj 'cosine' \
  --train_epochs 5 \
  --use_gpu True \
  --vali_interval 1\
  --gpu 0