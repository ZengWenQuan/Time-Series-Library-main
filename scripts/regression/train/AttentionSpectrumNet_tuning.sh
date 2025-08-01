#!/bin/bash

# 激活conda环境
conda activate mp

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 基础参数设置
BASE_CMD="python run.py \
  --task_name regression \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path data_stellar.csv \
  --model AttentionSpectrumNet \
  --feature_size 4802 \
  --label_size 4 \
  --train_epochs 100 \
  --patience 10 \
  --vali_interval 1 \
  --targets Teff logg FeH CFe \
  --split_ratio 0.7 0.15 0.15 \
  --use_gpu true \
  --gpu 0 \
  --gpu_type cuda \
  --loss SmoothL1 \
  --label_scaler_type standard \
  --features_scaler_type robust \
  --use_feh_sampling true \
  --feh_sampling_strategy balanced \
  --feh_index 2"

# ============ 测试不同的嵌入维度和注意力头数量 ============

# 嵌入维度和头数量组合
EMBED_DIMS=(64 128 256)
NUM_HEADS=(2 4 8)

# 使用固定的其他参数
OTHER_PARAMS="--patch_size 64 --stride 48 --conv_channels 32 64 128 --reduction_factor 4 --dropout_rate 0.2 --num_layers 3 --learning_rate 0.0005 --batch_size 64 --seed 42"

for embed_dim in "${EMBED_DIMS[@]}"; do
  for num_heads in "${NUM_HEADS[@]}"; do
    # 确保头数量是嵌入维度的因子
    if (( embed_dim % num_heads == 0 )); then
      model_id="AttentionSpectrumNet_embed${embed_dim}_heads${num_heads}"
      echo "运行配置: $model_id"
      $BASE_CMD --model_id $model_id --embed_dim $embed_dim --num_heads $num_heads $OTHER_PARAMS
    fi
  done
done

# ============ 测试不同的卷积通道数和内核大小 ============

# 使用最佳的嵌入维度和头数量
BEST_EMBED="--embed_dim 128 --num_heads 4"

# 卷积通道配置
CONV_CONFIGS=(
  "16 32 64"
  "32 64 128"
  "64 128 256"
)

# 卷积核大小配置
KERNEL_CONFIGS=(
  "3 5 7"
  "5 7 9"
  "3 7 11"
)

# 其他固定参数
OTHER_PARAMS="--patch_size 64 --stride 48 --reduction_factor 4 --dropout_rate 0.2 --num_layers 3 --learning_rate 0.0005 --batch_size 64 --seed 42"

for conv_config in "${CONV_CONFIGS[@]}"; do
  for kernel_config in "${KERNEL_CONFIGS[@]}"; do
    # 去除空格，用下划线替代，便于命名
    conv_name=$(echo $conv_config | tr ' ' '_')
    kernel_name=$(echo $kernel_config | tr ' ' '_')
    
    model_id="AttentionSpectrumNet_conv${conv_name}_kernel${kernel_name}"
    echo "运行配置: $model_id"
    
    $BASE_CMD --model_id $model_id $BEST_EMBED --conv_channels $conv_config --kernel_sizes $kernel_config $OTHER_PARAMS
  done
done

# ============ 测试不同的补丁大小和步长 ============

# 使用最佳的嵌入维度、头数量和卷积配置
BEST_EMBED="--embed_dim 128 --num_heads 4"
BEST_CONV="--conv_channels 32 64 128 --kernel_sizes 3 5 7"

# 补丁大小和步长配置
PATCH_SIZES=(32 64 96)
PATCH_STRIDES=(24 48 72)

# 其他固定参数
OTHER_PARAMS="--reduction_factor 4 --dropout_rate 0.2 --num_layers 3 --learning_rate 0.0005 --batch_size 64 --seed 42"

for patch_size in "${PATCH_SIZES[@]}"; do
  for stride in "${PATCH_STRIDES[@]}"; do
    # 确保步长小于等于补丁大小
    if (( stride <= patch_size )); then
      model_id="AttentionSpectrumNet_patch${patch_size}_stride${stride}"
      echo "运行配置: $model_id"
      $BASE_CMD --model_id $model_id $BEST_EMBED $BEST_CONV --patch_size $patch_size --stride $stride $OTHER_PARAMS
    fi
  done
done

# ============ 测试不同的学习率和批量大小 ============

# 使用最佳的模型架构参数
BEST_MODEL_PARAMS="--embed_dim 128 --num_heads 4 --conv_channels 32 64 128 --kernel_sizes 3 5 7 --patch_size 64 --stride 48 --reduction_factor 4 --dropout_rate 0.2 --num_layers 3"

# 学习率和批量大小配置
LEARNING_RATES=(0.0001 0.0005 0.001)
BATCH_SIZES=(32 64 128)

for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    model_id="AttentionSpectrumNet_lr${lr}_bs${bs}"
    echo "运行配置: $model_id"
    $BASE_CMD --model_id $model_id $BEST_MODEL_PARAMS --learning_rate $lr --batch_size $bs --seed 42
  done
done 