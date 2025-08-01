#!/bin/bash

# 激活conda环境
conda activate mp

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 基础命令
python run.py \
  --task_name regression \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path data_stellar.csv \
  --model_id AttentionSpectrumNet \
  --model AttentionSpectrumNet \
  --feature_size 4802 \
  --label_size 4 \
  --train_epochs 100 \
  --patience 10 \
  --vali_interval 1 \
  --seed 42 \
  --learning_rate 0.0005 \
  --batch_size 64 \
  --targets Teff logg FeH CFe \
  --split_ratio 0.7 0.15 0.15 \
  --use_gpu true \
  --gpu 0 \
  --gpu_type cuda \
  --embed_dim 128 \
  --num_heads 4 \
  --num_layers 3 \
  --patch_size 64 \
  --stride 48 \
  --conv_channels 32 64 128 \
  --kernel_sizes 3 5 7 \
  --reduction_factor 4 \
  --dropout_rate 0.2 \
  --loss SmoothL1 \
  --label_scaler_type standard \
  --features_scaler_type robust \
  --use_feh_sampling true \
  --feh_sampling_strategy balanced \
  --feh_index 2 