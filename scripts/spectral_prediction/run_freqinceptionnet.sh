#!/bin/bash

# 设置项目根目录 (如果需要，请根据您的环境修改)
export ROOT_PATH=$(pwd)

# 设置Python路径，以便能找到项目模块
export PYTHONPATH=$ROOT_PATH:$PYTHONPATH

# 定义模型和数据相关的变量
MODEL_NAME="FreqInceptionNet"
DATA_NAME="your_dataset_name" # <--- 在此填写您的数据集名称

# 定义训练会话的唯一标识符
RUN_ID="FreqInceptionNet_$(date +%Y%m%d_%H%M%S)"

# 训练参数
LEARNING_RATE=0.0001
BATCH_SIZE=32
EPOCHS=100

# 特征维度 (必须与YAML文件中的feature_size匹配)
FEATURE_SIZE=2000

# 标签维度 (必须与YAML文件中的label_size匹配)
LABEL_SIZE=4

# 数据文件路径 (请根据您的数据存放位置修改)
DATA_PATH="your_data.csv" # <--- 在此填写您的数据文件名

# 模型配置文件路径
MODEL_CONFIG_PATH="conf/freqinceptionnet.yaml"

# 日志和输出路径
LOG_FILE="./logs/${RUN_ID}.log"
CHECKPOINT_PATH="./checkpoints/${RUN_ID}"

# 创建必要的目录
mkdir -p ./logs
mkdir -p $CHECKPOINT_PATH

# 打印将要执行的命令
echo "Starting training for $MODEL_NAME with run ID: $RUN_ID"
echo "Configuration file: $MODEL_CONFIG_PATH"
echo "Checkpoints will be saved to: $CHECKPOINT_PATH"

# 执行训练命令
# 注意：这里的参数名（如--learning_rate）需要与您的run.py脚本接收的参数名完全一致
python -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --data_path $DATA_PATH \
  --model_id "${DATA_NAME}_${RUN_ID}" \
  --model $MODEL_NAME \
  --data $DATA_NAME \
  --features M \
  --label_len $LABEL_SIZE \
  --feature_size $FEATURE_SIZE \
  --model_conf $MODEL_CONFIG_PATH \
  --patience 20 \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --loss 'mse' \
  --lradj type1 \
  --gradient_clip_val 1.0 \
  --train_epochs $EPOCHS \
  --use_gpu True \
  --gpu 0 > $LOG_FILE 2>&1 &

# 提示用户训练已在后台开始
echo "Training started in background. Log file: $LOG_FILE"
echo "To view logs, run: tail -f $LOG_FILE"
echo "To check running processes, use: ps aux | grep run.py"
