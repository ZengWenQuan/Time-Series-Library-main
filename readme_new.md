# 光谱预测任务 (Spectral Prediction Task)

本项目扩展了Time Series Library框架，专门用于恒星光谱参数预测任务。该任务旨在通过分析恒星光谱数据来估计恒星的物理参数，包括有效温度(Teff)、表面重力(log g)、金属丰度(FeH)和碳增强度(CFe)。

## 📊 任务概述

### 任务类型
- **任务名称**: `spectral_prediction`
- **任务类型**: 多元回归 (Multi-variate Regression) 
- **输入**: 恒星光谱数据 (4802维连续谱 + 4802维归一化谱)
- **输出**: 4个恒星参数 [Teff, log g, FeH, CFe]

### 数据特点
- **输入维度**: 9604 (4802 continuum + 4802 normalized spectra)
- **输出维度**: 4 (stellar parameters)
- **数据集**: 基于LAMOST光谱巡天项目的恒星光谱数据
- **样本分布**: 训练集80% / 验证集10% / 测试集10%

## 🗂️ 项目结构

```
├── data_provider/
│   └── data_loader_spectral.py         # 光谱数据加载器
├── dataset/spectral/                   # 光谱数据集
│   ├── final_spectra_continuum.csv     # 连续谱数据
│   ├── final_spectra_normalized.csv    # 归一化光谱数据
│   └── removed_with_rv.csv             # 恒星参数标签
├── exp/
│   └── exp_spectral_prediction.py      # 光谱预测实验管理器  
├── models/spectral_prediction/         # 光谱预测模型库
│   ├── MLP.py                          # 多层感知机模型
│   ├── TwoBranchTeffNet.py            # 双分支Transformer网络
│   ├── SpectralMPBDNet.py             # 光谱MPBD网络
│   ├── DualPyramidNet.py              # 双金字塔网络 (新)
│   └── mspdownsample.py               # 多尺度下采样网络
├── conf/                              # 模型配置文件
│   ├── mlp.yaml
│   ├── twobranchteffnet.yaml
│   ├── dualpyramidnet.yaml            # 双金字塔网络配置
│   └── mspdownsample.yaml
└── scripts/spectral_prediction/        # 训练脚本
    ├── mlp.sh
    ├── twobranchteffnet.sh
    ├── dualpyramidnet.sh
    └── mpbdnet.sh
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装必要依赖
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib pyyaml scikit-learn
```

### 2. 数据准备
确保光谱数据文件位于正确位置：
```
dataset/spectral/
├── final_spectra_continuum.csv      # 连续谱特征
├── final_spectra_normalized.csv     # 归一化谱特征  
└── removed_with_rv.csv              # 目标参数标签
```

### 3. 模型训练
```bash
# 训练MLP模型
bash scripts/spectral_prediction/mlp.sh

# 训练双分支Transformer模型
bash scripts/spectral_prediction/twobranchteffnet.sh

# 训练双金字塔网络 (推荐)
bash scripts/spectral_prediction/dualpyramidnet.sh
```

### 4. 自定义训练
```bash
python run.py \
    --task_name spectral_prediction \
    --model DualPyramidNet \
    --model_id my_experiment \
    --is_training 1 \
    --data steller \
    --root_path ./dataset/spectral/ \
    --data_path removed_with_rv.csv \
    --spectra_continuum_path final_spectra_continuum.csv \
    --spectra_normalized_path final_spectra_normalized.csv \
    --label_path removed_with_rv.csv \
    --feature_size 4802 \
    --label_size 4 \
    --model_conf ./conf/dualpyramidnet.yaml \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 50 \
    --patience 10 \
    --use_gpu True
```

## 🧠 支持的模型

### 1. MLP (Multi-Layer Perceptron)
- **特点**: 简单的全连接网络，作为基线模型
- **配置**: `conf/mlp.yaml`
- **适用场景**: 快速原型验证，计算资源有限的情况

### 2. TwoBranchTeffNet 
- **特点**: 双分支Transformer架构，分别处理连续谱和归一化谱
- **配置**: `conf/twobranchteffnet.yaml`
- **适用场景**: 需要关注光谱序列特征的应用

### 3. SpectralMPBDNet
- **特点**: 基于多尺度patch的双分支网络
- **配置**: `conf/mpbdnet.yaml`
- **适用场景**: 多尺度特征提取

### 4. DualPyramidNet ⭐ (推荐)
- **特点**: 
  - 双金字塔特征提取器分别处理连续谱和归一化谱
  - 多尺度卷积金字塔捕获不同粒度的光谱特征
  - 注意力机制增强重要特征
  - 残差连接避免梯度消失
- **配置**: `conf/dualpyramidnet.yaml`
- **架构优势**:
  - 多尺度特征融合 (3, 5, 7 kernel sizes)
  - 自适应注意力权重
  - 层次化特征提取 [16→32→64 channels]
  - 双路径处理不同类型光谱数据

### 5. MSPDownsample
- **特点**: 多尺度下采样网络
- **配置**: `conf/mspdownsample.yaml`
- **适用场景**: 高维数据降维处理

## 📈 评估指标

### 回归指标
- **MAE (Mean Absolute Error)**: 平均绝对误差
- **MSE (Mean Squared Error)**: 均方误差  
- **RMSE (Root Mean Square Error)**: 均方根误差
- **R² Score**: 决定系数

### FeH分类指标
由于FeH参数具有离散分布特征，项目还提供基于FeH的分类评估：
- **Accuracy**: 分类准确率
- **Precision/Recall/F1**: 针对不同FeH区间的分类性能

## 🔧 配置说明

### 核心参数
- `feature_size`: 光谱特征维度 (默认4802)
- `label_size`: 目标参数数量 (默认4，对应Teff/log g/FeH/CFe)
- `split_ratio`: 数据集划分比例 (默认[0.8, 0.1, 0.1])
- `targets`: 目标参数名称 (默认['Teff', 'logg', 'FeH', 'CFe'])

### 模型特定配置
每个模型都有对应的YAML配置文件，定义网络架构和超参数。例如DualPyramidNet:

```yaml
pyramid_channels: [16, 32, 64]      # 金字塔通道数
kernel_sizes: [3, 5, 7]            # 多尺度卷积核
use_batch_norm: True               # 批归一化
use_attention: True                # 注意力机制
attention_reduction: 8             # 注意力降维比例  
fc_hidden_dims: [256, 128]         # 全连接层维度
dropout: 0.1                       # Dropout比例
```

## 📊 输出结果

训练完成后，结果保存在 `runs/spectral_prediction/ModelName/timestamp/` 目录下：

```
runs/spectral_prediction/DualPyramidNet/20250804_081428/
├── checkpoints/                    # 模型检查点
│   ├── best.pth                   # 最佳模型
│   └── last.pth                   # 最后epoch模型
├── metrics/                       # 评估指标
│   ├── best/                      # 最佳模型指标
│   └── last/                      # 最后模型指标
├── test_results/                  # 测试结果
│   └── predictions.csv            # 预测结果CSV
├── loss_curve.pdf                 # 损失曲线图
├── model.txt                      # 模型结构信息
├── training.log                   # 训练日志
└── scripts/                       # 使用的训练脚本
```

## ⚡ 性能优化

### 训练技巧
- **梯度裁剪**: 防止梯度爆炸 (`max_grad_norm=1.0`)
- **早停机制**: 避免过拟合 (`patience=10`)
- **学习率调度**: 自适应学习率调整
- **损失阈值**: 跳过异常高损失的批次 (`loss_threshold=100000.0`)

### 数据处理
- **特征标准化**: 连续谱数据标准化处理
- **多尺度输入**: 同时利用连续谱和归一化谱信息
- **批处理**: 高效的数据加载和批处理机制

## 🎯 应用场景

1. **天体物理研究**: 大规模恒星参数测量
2. **光谱分析**: 自动化光谱参数提取
3. **数据挖掘**: 从海量光谱数据中发现规律
4. **质量控制**: 光谱数据质量评估和筛选

## 🔍 进阶使用

### 自定义模型
参考现有模型实现，创建新的光谱预测模型：

```python
from exp.exp_basic import register_model

@register_model('MyModel')
class MySpectralModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        # ... 你的模型实现
    
    def forward(self, x_enc, **kwargs):
        # 处理输入光谱数据
        return predictions
```

### 自定义损失函数
在 `utils/losses.py` 中添加新的损失函数：

```python
class SpectralFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # 你的损失函数实现
        return loss
```

## 📚 相关资源

- **LAMOST数据发布**: http://www.lamost.org/
- **恒星参数标准**: IAU恒星参数定义
- **Time Series Library**: 基础时间序列建模框架

## 🤝 贡献指南

欢迎提交新的光谱预测模型和改进建议：

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/new-model`)
3. 提交更改 (`git commit -am 'Add new spectral model'`)
4. 推送分支 (`git push origin feature/new-model`)
5. 创建Pull Request

## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

*最后更新: 2025-08-04*