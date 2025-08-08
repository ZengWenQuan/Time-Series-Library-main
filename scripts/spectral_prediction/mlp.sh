

#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --split_data_path ./dataset/split_data \
  --continuum_filename continuum.csv \
  --normalized_filename normalized.csv \
  --labels_filename labels.csv \
  --model_id MLP_spectral \
  --stats_path conf/stats.yaml \
  --model MLP \
  --data spectral \
  --features M \
  --feature_size 4700 \
  --label_size 4 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10

