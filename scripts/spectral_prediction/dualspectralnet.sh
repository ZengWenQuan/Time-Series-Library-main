#!/bin/bash

# DualSpectralNet 双分支光谱预测模型训练脚本
# 
# 模型特点：
# - 连续谱分支：CNN+Transformer提取全局特征
# - 吸收线分支：多尺度CNN提取局部特征
# - 交叉注意力特征融合
# - 渐进式多尺度下采样

# 检查必要的环境
if [ ! -f "./run.py" ]; then
    echo "错误: 找不到 run.py 文件，请确保在正确的目录下运行此脚本"
    exit 1
fi

if [ ! -f "./conf/dualspectralnet.yaml" ]; then
    echo "错误: 找不到 dualspectralnet.yaml 配置文件"
    exit 1
fi

echo "=================================================="
echo "DualSpectralNet 双分支光谱预测模型训练"
echo "=================================================="

# 设置默认参数
MODEL_NAME="DualSpectralNet"
TASK_NAME="spectral_prediction"
DATA_NAME="spectral"
MODEL_ID="${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"

# 数据路径配置
ROOT_PATH="./dataset/spectral/"
DATA_PATH="removed_with_rv.csv"
SPECTRA_CONTINUUM_PATH="final_spectra_continuum.csv"
SPECTRA_NORMALIZED_PATH="final_spectra_normalized.csv"
LABEL_PATH="removed_with_rv.csv"

# 模型配置
MODEL_CONF="./conf/dualspectralnet.yaml"
FEATURE_SIZE=4700
LABEL_SIZE=4
TARGETS="Teff,logg,FeH,CFe"

# 训练参数
BATCH_SIZE=32
LEARNING_RATE=0.0001
TRAIN_EPOCHS=100
PATIENCE=15
RANDOM_SEED=2021

# GPU配置
USE_GPU=1
GPU_ID=0


echo "模型配置:"
echo "  模型名称: ${MODEL_NAME}"
echo "  任务类型: ${TASK_NAME}"
echo "  特征维度: ${FEATURE_SIZE}"
echo "  标签维度: ${LABEL_SIZE}"
echo "  目标参数: ${TARGETS}"
echo "  批次大小: ${BATCH_SIZE}"
echo "  学习率: ${LEARNING_RATE}"
echo "  训练轮数: ${TRAIN_EPOCHS}"
echo "  随机种子: ${RANDOM_SEED}"
echo ""

# 创建结果目录
mkdir -p "${RUN_DIR}"

echo "开始训练..."
echo ""

# 执行训练命令
python -u run.py \
  --task_name "${TASK_NAME}" \
  --is_training 1 \
  --model_id "${MODEL_ID}" \
  --model "${MODEL_NAME}" \
  --data "${DATA_NAME}" \
  --root_path "${ROOT_PATH}" \
  --data_path "${DATA_PATH}" \
  --spectra_continuum_path "${SPECTRA_CONTINUUM_PATH}" \
  --spectra_normalized_path "${SPECTRA_NORMALIZED_PATH}" \
  --label_path "${LABEL_PATH}" \
  --feature_size ${FEATURE_SIZE} \
  --label_size ${LABEL_SIZE} \
  --model_conf "${MODEL_CONF}" \
  
  --split_ratio 0.8,0.1,0.1 \
  --features_scaler_type "standard" \
  --label_scaler_type "standard" \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --train_epochs ${TRAIN_EPOCHS} \
  --patience ${PATIENCE} \
  --use_gpu ${USE_GPU} \
  --gpu ${GPU_ID} \
  --des "DualSpectralNet双分支光谱预测模型训练" \
  --dropout 0.1 \
  --seed ${RANDOM_SEED}

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "训练完成!"
    echo "=================================================="
    echo "结果保存在: ${RUN_DIR}"
    echo ""
    echo "进行测试评估..."
    
    # 执行测试
    python -u run.py \
      --task_name "${TASK_NAME}" \
      --is_training 0 \
      --model_id "${MODEL_ID}" \
      --model "${MODEL_NAME}" \
      --data "${DATA_NAME}" \
      --root_path "${ROOT_PATH}" \
      --data_path "${DATA_PATH}" \
      --spectra_continuum_path "${SPECTRA_CONTINUUM_PATH}" \
      --spectra_normalized_path "${SPECTRA_NORMALIZED_PATH}" \
      --label_path "${LABEL_PATH}" \
      --feature_size ${FEATURE_SIZE} \
      --label_size ${LABEL_SIZE} \
      --model_conf "${MODEL_CONF}" \
      
      --split_ratio 0.8,0.1,0.1 \
      --features_scaler_type "standard" \
      --label_scaler_type "standard" \
      --batch_size ${BATCH_SIZE} \
      --use_gpu ${USE_GPU} \
      --gpu ${GPU_ID} \
      --run_dir "${RUN_DIR}" \
      --des "DualSpectralNet模型测试评估" \
      --dropout 0.1 \
      --seed ${RANDOM_SEED}
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================="
        echo "测试完成!"
        echo "=================================================="
        echo "测试结果保存在: ${RUN_DIR}"
        
        # 显示结果文件
        echo ""
        echo "生成的文件:"
        ls -la "${RUN_DIR}/"
        
        # 如果有测试结果，显示性能指标
        if [ -f "${RUN_DIR}/result.txt" ]; then
            echo ""
            echo "测试结果摘要:"
            cat "${RUN_DIR}/result.txt"
        fi
    else
        echo "测试过程中出现错误"
        exit 1
    fi
else
    echo ""
    echo "训练过程中出现错误"
    exit 1
fi

echo ""
echo "=================================================="
echo "DualSpectralNet 训练和测试流程完成"
echo "=================================================="