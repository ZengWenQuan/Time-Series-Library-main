#!/bin/bash

python -u run.py \
  --task_name stellar_parameter_estimation \
  --is_training 1 \
  --model_id stellar_inception1d \
  --model Inception1DModel \
  --data stellar \
  --root_path ./dataset/stellar/ \
  --data_path merged_output_A.csv \
  --features S \
  --feature_size 4802 \
  --label_size 4 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 4 \
  --c_out 4 \
  --d_model 64 \
  --d_ff 128 \
  --n_heads 4 \
  --batch_size 8 \
  --dropout 0.2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 30 \
  --patience 5 