#!/bin/bash

# 示例训练脚本，展示如何使用我们的新 DualBranchSpectralModel

echo "Training DualBranchSpectralModel with AblativeDualEncoder backbone..."

python run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --model_id dual_branch_test \
  --model DualBranchSpectralModel \
  --model_conf ./conf/dualbranchspectral.yaml \
  --data stellar \
  --root_path ./dataset/split_data \
  --stats_path conf/stats.yaml \
  --feature_size 4704 \
  --label_size 4 \
  --train_epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --use_gpu True \
  --gpu 0

echo "Training completed!"