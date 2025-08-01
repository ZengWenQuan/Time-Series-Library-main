#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 # 设置GPU编号

# 激活conda环境
source ~/.bashrc
source activate mp

# 基本参数设置
data_path="merged_output.csv"
model_name="HybridModel"
base_model_id="hybrid_tuning"
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

# 损失函数选项
loss_functions=("MSE" "SmoothL1" "FocalSmoothL1" "Huber" "LogCosh")

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

# 全连接层大小选项
ffn_hidden_size_options=(64 128 256)

# Dropout率选项
dropout_rate_options=(0.1 0.3 0.5)

# Focal Loss参数选项
focal_alpha_options=(0.5 1.0 2.0)
focal_gamma_options=(1.0 2.0 3.0)
focal_threshold_options=(0.3 0.5 1.0)

# 学习率选项
learning_rate_options=(0.001 0.0005 0.0001)

# 批量大小选项
batch_size_options=(16 32 64)

# 调参循环
# 注意：完整的网格搜索会产生非常多的组合，这里只选择部分组合进行演示
# 实际使用时可以根据需要调整循环范围

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
              --loss FocalSmoothL1 \
              --focal_alpha 1.0 \
              --focal_gamma 2.0 \
              --focal_threshold 0.5
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
      --loss FocalSmoothL1 \
      --focal_alpha 1.0 \
      --focal_gamma 2.0 \
      --focal_threshold 0.5
done

# 循环3：尝试不同的损失函数和学习率调度器
for loss_function in "${loss_functions[@]}"; do
    for lr_scheduler in "${lr_schedulers[@]}"; do
        model_id="${base_model_id}_loss_${loss_function}_lr_${lr_scheduler}"
        echo "运行配置: 损失函数=${loss_function}, 学习率调度器=${lr_scheduler}"
        
        # 对于Focal损失函数，添加相应的参数
        focal_params=""
        if [[ $loss_function == Focal* ]]; then
            focal_params="--focal_alpha 1.0 --focal_gamma 2.0 --focal_threshold 0.5"
        fi
        
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
          --loss $loss_function \
          $focal_params
    done
done

# 循环4：尝试不同的Focal Loss参数（仅当使用Focal Loss时）
for alpha in "${focal_alpha_options[@]}"; do
    for gamma in "${focal_gamma_options[@]}"; do
        for threshold in "${focal_threshold_options[@]}"; do
            model_id="${base_model_id}_focal_${alpha}_${gamma}_${threshold}"
            echo "运行配置: Focal Loss参数 alpha=${alpha}, gamma=${gamma}, threshold=${threshold}"
            
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
              --lradj warmup_cosine \
              --loss FocalSmoothL1 \
              --focal_alpha $alpha \
              --focal_gamma $gamma \
              --focal_threshold $threshold
        done
    done
done

# 循环5：尝试不同的学习率和批量大小
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
          --loss FocalSmoothL1 \
          --focal_alpha 1.0 \
          --focal_gamma 2.0 \
          --focal_threshold 0.5
    done
done

echo "参数调优完成！" 