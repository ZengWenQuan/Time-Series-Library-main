#!/bin/bash

# Script to train the MPBDNet model for spectral parameter prediction.

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum.csv \
  --spectra_normalized_path final_spectra_normalized.csv \
  --label_path removed_with_rv.csv \
  --stats_path conf/stats.yaml \
  \
  --model_id SpectralMPBDNet_spectral \
  --model SpectralMPBDNet \
  --model_conf conf/mpbdnet.yaml \
  \
  --data spectral \
  --features M \
  --feature_size 4700 \
  --label_size 4 \
  \
  --des 'Training MPBDNet model' \
  --itr 1 \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --dropout 0.3 \
  --patience 5 \
  --loss_threshold 100000.0 \
  --max_grad_norm 1.0

