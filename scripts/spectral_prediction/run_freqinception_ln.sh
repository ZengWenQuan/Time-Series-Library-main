#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum_full.csv \
  --spectra_normalized_path final_spectra_normalized_full.csv \
  --label_path removed_with_rv.csv \
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