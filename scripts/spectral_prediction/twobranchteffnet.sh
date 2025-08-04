#!/bin/bash

# Script to train the TwoBranchTeffNet model for spectral parameter prediction.

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum.csv \
  --spectra_normalized_path final_spectra_normalized.csv \
  --label_path removed_with_rv.csv \
  --stats_path conf/stats.yaml \
  \
  --model_id TwoBranchTeffNet_spectral \
  --model TwoBranchTeffNet \
  --model_conf conf/twobranchteffnet.yaml \
  \
  --data spectral \
  --features M \
  --feature_size 4700 \
  --label_size 4 \
  --targets "['Teff', 'logg', 'FeH', 'CFe']" \
  \
  --des 'Training TwoBranchTeffNet model' \
  --itr 1 \
  --train_epochs 100 \
  --batch_size 64 \
  --learning_rate 0.00005 \
  --dropout 0.2 \
  --patience 70 \
  --loss_threshold 100000.0 \
  --max_grad_norm 5.0\

