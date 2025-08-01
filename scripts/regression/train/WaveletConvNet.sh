#!/bin/bash

# 激活conda环境
conda activate mp

# 设置模型和数据参数
model_name="WaveletConvNet"
data_path="merged_output_A.csv"  # 替换为实际的数据文件名
model_id="wavelet_test"

# 运行训练
python run.py \
  --task_name stellar_parameter_estimation \
  --is_training 1 \
  --model $model_name \
  --data stellar \
  --root_path ./dataset/stellar/ \
  --data_path $data_path \
  --model_id $model_id \
  --features M \
  --feature_size 4802 \
  --label_size 4 \
  --enc_in 4802 \
  --dec_in 4802 \
  --c_out 4 \
  --d_model 4802 \
  --d_ff 128 \
  --e_layers 1 \
  --batch_size 8 \
  --learning_rate 0.0005 \
  --train_epochs 10 \
  --patience 5 \
  --vali_interval 2 \
  --dwt_level 3 \
  --wavelet db1 \
  --ffn_dim 128 \
  --gpu 0 

  