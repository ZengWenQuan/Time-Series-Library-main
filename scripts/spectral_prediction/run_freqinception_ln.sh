#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum.csv \
  --spectra_normalized_path final_spectra_normalized.csv \
  --label_path removed_with_rv.csv \
  --model_id FreqInceptionLNet_spectral \
  --model FreqInceptionLNet \
  --data spectral \
  --features M \
  --feature_size 2000 \
  --label_size 4 \
  --stats_path conf/stats.yaml \
  --model_conf conf/freqinception_ln.yaml \
  --patience 2 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --loss 'mse' \
  --lradj 'cosine' \
  --train_epochs 10 \
  --use_gpu True \
  --gpu 0