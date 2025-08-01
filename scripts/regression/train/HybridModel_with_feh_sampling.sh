#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 # 设置GPU编号

# 激活conda环境
source ~/.bashrc
source activate mp

# 设置模型和数据参数
model_name="HybridModel"
data_path="merged_output.csv"
model_id="train_sfft_hybrid_feh_sampling"
train_epochs=500
patience=5000
vali_interval=5
gpu=0

# 运行训练
python run.py \
  --task_name regression \
  --is_training 1 \
  --model $model_name \
  --data stellar \
  --root_path ./dataset/stellar/ \
  --data_path $data_path \
  --model_id $model_id \
  --plot_loss False \
  --features M \
  --feature_size 4802 \
  --label_size 4 \
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
  --label_scaler_type standard \
  --apply_inverse_transform 1 \
  --lradj warmup_cosine \
  --loss SmoothL1 \
  --use_feh_sampling \
  --feh_sampling_strategy balanced \
  --feh_sampling_k_neighbors 5 \
  --feh_index 2 \
  --seed 42 