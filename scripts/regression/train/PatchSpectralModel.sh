#!/bin/bash

python -u run.py \
  --task_name stellar_parameter_estimation \
  --is_training 1 \
  --model_id stellar_patchspectral_v2 \
  --model PatchSpectralModel \
  --data_set_name stellar \
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
  --d_model 16 \
  --d_ff 32 \
  --n_heads 2 \
  --batch_size 4 \
  --dropout 0.2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 50 \
  --patience 10 