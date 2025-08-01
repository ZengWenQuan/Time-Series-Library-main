# SFFTDualBranchNet - SFFT双分支网络

## 模型概述

SFFTDualBranchNet是一个专为回归任务设计的深度学习模型，特别适用于光谱数据分析和恒星参数估计。该模型采用创新的双分支架构，结合短时傅里叶变换(SFFT)特征提取和多尺度卷积处理。

## 模型架构

### 1. SFFT特征提取器 (SFFTFeatureExtractor)
- **功能**: 将一维输入序列转换为二维时频特征图
- **方法**: 使用短时傅里叶变换(STFT)提取频域特征
- **输出**: 幅度谱特征图 (频率 × 时间)

### 2. 双分支处理

#### 分支A: 全卷积分支 (FullConvBranch)
- **结构**: 5层卷积神经网络
- **特点**: 
  - 每层包含卷积、批归一化、ReLU激活
  - 3×3平均池化进行下采样
  - 通道数逐层递增 (64, 128, 256, 512, 1024)
- **作用**: 提取全局特征和深层语义信息

#### 分支B: Inception分支 (InceptionBranch)
- **结构**: 5层Inception风格卷积层
- **特点**:
  - 每层使用多种卷积核尺寸 (1×1, 3×3, 5×5, 7×7)
  - 自适应padding保持特征图尺寸一致
  - 通道维度拼接多尺度特征
  - 3×3平均池化下采样
- **作用**: 捕获多尺度特征和局部模式

### 3. 特征融合与输出
- **融合**: 将两个分支的特征展平后拼接
- **FFN**: 三层前馈神经网络
  - 输入层 → 隐藏层(512) → 隐藏层(256) → 输出层
  - ReLU激活和Dropout正则化
- **输出**: 回归预测结果

## 模型特点

### 优势
1. **多尺度特征提取**: Inception分支捕获不同尺度的特征模式
2. **频域信息利用**: SFFT提取频域特征，增强模型对周期性和频率特征的感知
3. **双分支互补**: 全卷积分支提供全局特征，Inception分支提供局部多尺度特征
4. **参数高效**: 合理的网络设计平衡了模型复杂度和性能

### 适用场景
- 光谱数据分析
- 恒星参数估计 (有效温度、表面重力、金属丰度等)
- 时间序列回归
- 信号处理相关的回归任务

## 使用方法

### 1. 模型配置
```python
class Config:
    def __init__(self):
        self.feature_size = 1024  # 输入序列长度
        self.label_size = 3       # 输出标签数量
```

### 2. 模型实例化
```python
from models.regression.SFFTDualBranchNet import Model

config = Config()
model = Model(config)
```

### 3. 训练示例
```bash
# 使用提供的训练脚本
bash scripts/regression/train/SFFTDualBranchNet_example.sh
```

### 4. 推理示例
```python
import torch

# 准备输入数据
input_data = torch.randn(batch_size, feature_size)

# 模型推理
model.eval()
with torch.no_grad():
    predictions = model(input_data)
```

## 参数配置

### 必需参数
- `feature_size`: 输入特征维度
- `label_size`: 输出标签维度

### 可调参数
- SFFT参数: `n_fft`, `hop_length`, `win_length`
- 卷积分支基础通道数: `base_channels`
- FFN隐藏层维度和dropout率

## 性能特征

### 模型规模
- 参数数量: ~72M (取决于输入尺寸)
- 内存占用: 适中
- 计算复杂度: O(n log n) (SFFT) + O(n²) (卷积)

### 训练建议
- 学习率: 0.001 (Adam优化器)
- 批大小: 32-64
- 早停耐心值: 10-20 epochs
- 数据标准化: 推荐使用StandardScaler

## 注意事项

1. **输入数据要求**: 一维序列数据，长度需要足够支持SFFT变换
2. **内存使用**: SFFT会增加内存使用，注意批大小设置
3. **数据预处理**: 建议对输入特征和标签都进行标准化
4. **GPU加速**: 模型支持GPU训练，推荐使用CUDA

## 扩展性

模型设计具有良好的扩展性：
- 可以调整分支数量和结构
- 支持不同的特征提取方法
- 可以添加注意力机制
- 支持多任务学习扩展

## 引用

如果您在研究中使用了此模型，请引用相关工作。