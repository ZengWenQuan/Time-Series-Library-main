

#!/bin/bash

python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --data_path non_existent.csv \
  --model_id MLP_spectral \
  --model MLP \
  --data spectral \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 4 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9604 \
  --dec_in 9604 \
  --c_out 4 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10

