#!/bin/bash

# 激活conda环境
conda activate mp

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 选择最佳模型进行测试
# 注意: 实际测试时，需要选择训练表现最好的模型路径
# 这里假设使用的是AttentionSpectrumNet模型的最佳配置

python run.py \
  --task_name regression \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path data_stellar.csv \
  --model_id AttentionSpectrumNet \
  --model AttentionSpectrumNet \
  --feature_size 4802 \
  --label_size 4 \
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
  --label_scaler_type standard \
  --features_scaler_type robust \
  --calculate_feh_metrics 1 