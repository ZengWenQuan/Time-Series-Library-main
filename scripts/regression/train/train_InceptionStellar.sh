#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 # 设置GPU编号

# 激活conda环境
conda activate mp

# 设置模型和数据参数
model_name="InceptionStellar"
data_path="merged_output.csv"  # 替换为实际的数据文件名
model_id="train_metrics"
train_epochs=200
patience=500
vali_interval=5
gpu=0
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
  --d_model 512 \
  --d_ff 512 \
  --e_layers 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs $train_epochs \
  --patience $patience \
  --vali_interval $vali_interval \
  --gpu $gpu 