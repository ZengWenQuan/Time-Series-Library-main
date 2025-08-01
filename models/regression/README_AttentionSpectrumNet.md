# AttentionSpectrumNet：结合注意力机制和卷积网络的恒星参数估计模型

## 模型概述

AttentionSpectrumNet 是一个专为恒星光谱数据分析设计的深度学习模型，它结合了卷积神经网络的局部特征提取能力和自注意力机制的全局依赖性捕获能力。该模型特别适合处理高维光谱数据，通过将光谱分割成重叠补丁并分别处理，有效降低了计算复杂度并提高了对关键光谱特征的敏感性。

## 模型架构

AttentionSpectrumNet 的核心架构包括以下组件：

1. **补丁提取层**：将原始光谱数据（长度为4802）分割成多个重叠的短序列补丁（默认长度为64，步长为48）
2. **多尺度卷积特征提取**：使用不同大小的卷积核（如3×1, 5×1, 7×1）从每个补丁中提取多尺度特征
3. **特征降维**：通过线性投影降低特征维度，减少后续注意力计算的复杂度
4. **位置编码**：添加正余弦位置编码，为注意力机制提供位置信息
5. **自注意力层**：使用多头自注意力机制捕获补丁之间的相互关系
6. **分类令牌**：类似于BERT/ViT中的CLS令牌，用于聚合所有补丁的信息
7. **预测头**：基于分类令牌的表示进行恒星参数（Teff, logg, FeH, CFe）的回归预测

## 主要特点

- **补丁处理**：通过将长序列分割成多个补丁，有效处理高维光谱数据，降低计算复杂度
- **多尺度特征提取**：利用不同大小的卷积核捕获不同尺度的光谱特征
- **注意力机制**：使用自注意力机制建模光谱不同部分之间的长距离依赖关系
- **位置感知**：通过位置编码保留光谱中的位置信息
- **端到端训练**：从原始光谱直接预测恒星参数，无需手工特征工程

## 使用方法

### 训练模型

```bash
# 基础训练命令
bash scripts/regression/train/AttentionSpectrumNet.sh

# 或者直接使用Python命令
python run.py \
  --task_name regression \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path data_stellar.csv \
  --model AttentionSpectrumNet \
  --feature_size 4802 \
  --label_size 4 \
  --targets Teff logg FeH CFe
```

### 超参数调优

```bash
# 调优不同的模型配置
bash scripts/regression/train/AttentionSpectrumNet_tuning.sh
```

### 重要超参数说明

- `embed_dim`: 注意力层的嵌入维度（默认128）
- `num_heads`: 多头注意力的头数量（默认4）
- `num_layers`: 注意力层的数量（默认3）
- `patch_size`: 每个补丁的大小（默认64）
- `stride`: 补丁提取的步长（默认48）
- `conv_channels`: 卷积层的通道数列表（默认[32, 64, 128]）
- `kernel_sizes`: 卷积核大小列表（默认[3, 5, 7]）
- `reduction_factor`: 特征降维因子（默认4）

## 推荐配置

根据大量实验，我们推荐以下配置用于恒星参数估计任务：

```
--embed_dim 128 \
--num_heads 4 \
--num_layers 3 \
--patch_size 64 \
--stride 48 \
--conv_channels 32 64 128 \
--kernel_sizes 3 5 7 \
--reduction_factor 4 \
--dropout_rate 0.2 \
--loss SmoothL1
```

## 金属丰度（FeH）采样策略

由于恒星样本中金属丰度（FeH）分布不均衡，模型内置了FeH采样策略，通过过采样稀有类别和欠采样常见类别，使模型能更好地处理金属贫星。使用以下参数开启此功能：

```
--use_feh_sampling true \
--feh_sampling_strategy balanced
```

## 模型性能

在恒星光谱数据上的测试表明，AttentionSpectrumNet 在各项恒星参数预测上都取得了良好的性能，尤其在金属丰度（FeH）预测方面表现突出，对极度贫金属星和贫金属星的预测准确度显著提高。 