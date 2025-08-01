#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 # 设置GPU编号

# 激活conda环境
source ~/.bashrc
source activate mp

# 基本参数设置
data_path="merged_output.csv"
model_name="HybridModel"
base_model_id="hybrid_tuning_2"
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

# 学习率调度器选项
lr_schedulers=("cosine" "warmup_cosine" "type1" "type3")

# 损失函数选项（移除了FocalSmoothL1）
loss_functions=("MSE" "SmoothL1" "Huber" "MAE" "LogCosh")

# 全连接层大小选项
ffn_hidden_size_options=(64 128 256)

# Dropout率选项
dropout_rate_options=(0.1 0.3 0.5)

echo "开始执行调参脚本2：损失函数和学习率调度器调优"

# 循环1：尝试不同的损失函数和学习率调度器
for loss_function in "${loss_functions[@]}"; do
    for lr_scheduler in "${lr_schedulers[@]}"; do
        model_id="${base_model_id}_loss_${loss_function}_lr_${lr_scheduler}"
        echo "运行配置: 损失函数=${loss_function}, 学习率调度器=${lr_scheduler}"
        
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
          --label_scaler_type $label_scaler_type \
          --apply_inverse_transform $apply_inverse_transform \
          --lradj $lr_scheduler \
          --loss $loss_function
    done
done

# 循环2：尝试不同的全连接层大小
for ffn_size in "${ffn_hidden_size_options[@]}"; do
    model_id="${base_model_id}_ffn_${ffn_size}"
    echo "运行配置: 全连接层大小=${ffn_size}"
    
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
      --ffn_hidden_size $ffn_size \
      --dropout_rate 0.3 \
      --label_scaler_type $label_scaler_type \
      --apply_inverse_transform $apply_inverse_transform \
      --lradj warmup_cosine \
      --loss SmoothL1
done

# 循环3：尝试不同的Dropout率
for dropout in "${dropout_rate_options[@]}"; do
    model_id="${base_model_id}_dropout_${dropout}"
    echo "运行配置: Dropout率=${dropout}"
    
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
      --dropout_rate $dropout \
      --label_scaler_type $label_scaler_type \
      --apply_inverse_transform $apply_inverse_transform \
      --lradj warmup_cosine \
      --loss SmoothL1
done

echo "调参脚本2执行完成！" 