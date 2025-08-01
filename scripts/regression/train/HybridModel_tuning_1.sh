#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 # 设置GPU编号

# 激活conda环境
source ~/.bashrc
source activate mp

# 基本参数设置
data_path="merged_output.csv"
model_name="HybridModel"
base_model_id="hybrid_tuning_1"
task_name="regression"
feature_size=4802
label_size=4
# 删除targets和split_ratio变量
label_scaler_type="standard"
apply_inverse_transform=1
train_epochs=200
patience=50
vali_interval=5
gpu=0

# SFFT参数选项
n_fft_options=(128 256 512)
hop_length_options=(32 64 128)

# 卷积通道数选项
conv_channel_options=(
    "8 16 32"    # 小型网络
    "16 32 64"   # 中型网络
    "32 64 128"  # 大型网络
)

# Inception通道数选项
inception_channel_options=(
    "8 16 32"    # 小型网络
    "16 32 64"   # 中型网络
    "32 64 128"  # 大型网络
)

echo "开始执行调参脚本1：SFFT参数和网络结构调优"

# 循环1：尝试不同的SFFT参数
for n_fft in "${n_fft_options[@]}"; do
    for hop_length in "${hop_length_options[@]}"; do
        # 确保hop_length不大于n_fft的一半
        if (( hop_length <= n_fft/2 )); then
            model_id="${base_model_id}_sfft_${n_fft}_${hop_length}"
            echo "运行配置: SFFT参数 n_fft=${n_fft}, hop_length=${hop_length}"
            
            python run.py \
              --task_name $task_name \
              --is_training 1 \
              --model $model_name \
              --data stellar \
              --root_path ./dataset/stellar/ \
              --data_path $data_path \
              --model_id $model_id \
              --plot_loss False \
              --features M \
              --feature_size $feature_size \
              --label_size $label_size \
              --batch_size 32 \
              --learning_rate 0.0001 \
              --train_epochs $train_epochs \
              --patience $patience \
              --vali_interval $vali_interval \
              --gpu $gpu \
              --n_fft $n_fft \
              --hop_length $hop_length \
              --conv_channel_1 16 \
              --conv_channel_2 32 \
              --conv_channel_3 64 \
              --inception_channel_1 16 \
              --inception_channel_2 32 \
              --inception_channel_3 64 \
              --pool_size 2 \
              --ffn_hidden_size 128 \
              --dropout_rate 0.3 \
              --label_scaler_type $label_scaler_type \
              --apply_inverse_transform $apply_inverse_transform \
              --lradj warmup_cosine \
              --loss SmoothL1
        fi
    done
done

# 循环2：尝试不同的卷积和Inception通道配置
for ((i=0; i<${#conv_channel_options[@]}; i++)); do
    conv_channels=(${conv_channel_options[$i]})
    inception_channels=(${inception_channel_options[$i]})
    
    model_id="${base_model_id}_channels_${conv_channels[0]}_${inception_channels[0]}"
    echo "运行配置: 通道配置 conv=${conv_channels[*]}, inception=${inception_channels[*]}"
    
    python run.py \
      --task_name $task_name \
      --is_training 1 \
      --model $model_name \
      --data stellar \
      --root_path ./dataset/stellar/ \
      --data_path $data_path \
      --model_id $model_id \
      --plot_loss False \
      --features M \
      --feature_size $feature_size \
      --label_size $label_size \
      --batch_size 32 \
      --learning_rate 0.0001 \
      --train_epochs $train_epochs \
      --patience $patience \
      --vali_interval $vali_interval \
      --gpu $gpu \
      --n_fft 256 \
      --hop_length 64 \
      --conv_channel_1 ${conv_channels[0]} \
      --conv_channel_2 ${conv_channels[1]} \
      --conv_channel_3 ${conv_channels[2]} \
      --inception_channel_1 ${inception_channels[0]} \
      --inception_channel_2 ${inception_channels[1]} \
      --inception_channel_3 ${inception_channels[2]} \
      --pool_size 2 \
      --ffn_hidden_size 128 \
      --dropout_rate 0.3 \
      --label_scaler_type $label_scaler_type \
      --apply_inverse_transform $apply_inverse_transform \
      --lradj warmup_cosine \
      --loss SmoothL1
done

echo "调参脚本1执行完成！" 