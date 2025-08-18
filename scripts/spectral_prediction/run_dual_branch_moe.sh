#!/bin/bash
#
# DualBranchMoENet 双分支混合专家模型训练脚本
#
# 该脚本用于启动 DualBranchMoENet 模型的训练和评估流程。
# 它通过 run.py 调用核心实验框架，并传入所有必要的参数，
# 包括模型名称、配置文件、数据路径和训练超参数。

MODEL_NAME="DualBranchMoENet"
TASK_NAME="spectral_prediction"
DATA_NAME="spectral"
MODEL_ID="${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

SPLIT_DATA_PATH="./dataset/split_data"
CONTINUUM_FILENAME="continuum.csv"
NORMALIZED_FILENAME="normalized.csv"
LABELS_FILENAME="labels.csv"

MODEL_CONF="./conf/dual_branch_moe.yaml"
FEATURE_SIZE=4800
LABEL_SIZE=4
TARGETS="['Teff','logg','FeH','CFe']" #必须不能留空格
split_ratio='[0.8,0.2,0]' #必须不能留空格
TRAIN_EPOCHS=355
BATCH_SIZE=32
PATIENCE=10
LEARNING_RATE=0.0001
LR_ADJUST='cos'

USE_GPU=True
GPU_ID=0

python -u run.py \
  --task_name ${TASK_NAME} \
  --is_training 1 \
  --model_id ${MODEL_ID} \
  --model ${MODEL_NAME} \
  --split_ratio ${split_ratio}\
  --data ${DATA_NAME} \
  --model_conf ${MODEL_CONF} \
  --root_path ${SPLIT_DATA_PATH} \
  --continuum_filename ${CONTINUUM_FILENAME} \
  --normalized_filename ${NORMALIZED_FILENAME} \
  --labels_filename ${LABELS_FILENAME} \
  --feature_size ${FEATURE_SIZE} \
  --label_size ${LABEL_SIZE} \
  --stats_path conf/stats.yaml \
  
  --train_epochs ${TRAIN_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --patience ${PATIENCE} \
  --learning_rate ${LEARNING_RATE} \
  --lradj ${LR_ADJUST} \
  --use_gpu ${USE_GPU} \
  --use_amp \
  --gpu ${GPU_ID} \
  --des "${MODEL_NAME}_experiment"