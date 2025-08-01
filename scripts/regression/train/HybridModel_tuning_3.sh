#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 # 设置GPU编号

# 激活conda环境
source ~/.bashrc
source activate mp

# 基本参数设置
data_path="merged_output.csv"
model_name="HybridModel"
base_model_id="hybrid_tuning_3"
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

# 学习率选项
learning_rate_options=(0.001 0.0005 0.0001 0.00005 0.00001)

# 批量大小选项
batch_size_options=(16 32 64 128)

# 池化大小选项
pool_size_options=(1 2 3)

echo "开始执行调参脚本3：学习率、批量大小和其他超参数调优"

# 循环1：尝试不同的学习率和批量大小
for lr in "${learning_rate_options[@]}"; do
    for bs in "${batch_size_options[@]}"; do
        model_id="${base_model_id}_lr_${lr}_bs_${bs}"
        echo "运行配置: 学习率=${lr}, 批量大小=${bs}"
        
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
          --batch_size $bs \
          --learning_rate $lr \
          --train_epochs $train_epochs \
          --patience $patience \
          --vali_interval $vali_interval \
          --gpu $gpu \
          --n_fft 256 \
          --hop_length 64 \
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
    done
done

# 循环2：尝试不同的池化大小
for pool_size in "${pool_size_options[@]}"; do
    model_id="${base_model_id}_pool_${pool_size}"
    echo "运行配置: 池化大小=${pool_size}"
    
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
      --conv_channel_1 16 \
      --conv_channel_2 32 \
      --conv_channel_3 64 \
      --inception_channel_1 16 \
      --inception_channel_2 32 \
      --inception_channel_3 64 \
      --pool_size $pool_size \
      --ffn_hidden_size 128 \
      --dropout_rate 0.3 \
      --label_scaler_type $label_scaler_type \
      --apply_inverse_transform $apply_inverse_transform \
      --lradj warmup_cosine \
      --loss SmoothL1
done

# 循环3：尝试不同的标签缩放类型
for scaler_type in "standard" "minmax" "robust"; do
    model_id="${base_model_id}_scaler_${scaler_type}"
    echo "运行配置: 标签缩放类型=${scaler_type}"
    
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
      --conv_channel_1 16 \
      --conv_channel_2 32 \
      --conv_channel_3 64 \
      --inception_channel_1 16 \
      --inception_channel_2 32 \
      --inception_channel_3 64 \
      --pool_size 2 \
      --ffn_hidden_size 128 \
      --dropout_rate 0.3 \
      --label_scaler_type $scaler_type \
      --apply_inverse_transform 1 \
      --lradj warmup_cosine \
      --loss SmoothL1
done

echo "调参脚本3执行完成！" 