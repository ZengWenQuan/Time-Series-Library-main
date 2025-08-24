#!/bin/bash

# 脚本用于训练FreqInceptionConvNet模型，参数结构参考了项目中的其他标准脚本。

python3 -u run.py \
  --task_name spectral_prediction \
  --is_training 1 \
  --root_path ./dataset/spectral/ \
  --spectra_continuum_path final_spectra_continuum.csv \
  --spectra_normalized_path final_spectra_normalized.csv \
  --label_path removed_with_rv.csv \
  --model_id FreqInceptionConvNet_spectral_v1 \
  --model FreqInceptionConvNet \
  --data spectral \
  --features M \
  --feature_size 4700 \
  --label_size 4 \
  --stats_path conf/stats.yaml \
  --model_conf conf/freqinceptionconvnet.yaml \
  --patience 20000 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --train_epochs 5 \
  --use_gpu True \
  --vali_interval 1 \
  --gpu 0

