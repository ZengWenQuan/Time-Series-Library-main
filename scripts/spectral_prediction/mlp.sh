

#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum.csv \
  --spectra_normalized_path final_spectra_normalized.csv \
  --label_path removed_with_rv.csv \
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

