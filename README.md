
# 恒星光谱参数预测任务 (Stellar Spectral Parameter Prediction Task)

本项目是一个专注于恒星光谱参数预测的深度学习框架。其核心任务是通过分析恒星光谱数据，精准地估计恒星的关键物理参数，包括有效温度(Teff)、表面重力(log g)、金属丰度([Fe/H])和碳丰度([C/Fe])。

## 📊 任务概述

- **任务名称**: `spectral_prediction`
- **任务类型**: 多元回归 (Multi-variate Regression)
- **输入**: 恒星光谱数据 (通常分为连续谱和归一化谱两个分支)
- **输出**: 4个核心恒星参数: `[Teff, logg, FeH, CFe]`

## 🗂️ 项目结构

```
├── data_provider/
│   └── data_loader_spectral.py    # 核心光谱数据加载器
├── dataset/spectral/              # 光谱数据集存放目录
│   ├── final_spectra_continuum.csv
│   ├── final_spectra_normalized.csv
│   └── removed_with_rv.csv
├── exp/
│   └── exp_spectral_prediction.py # 光谱预测任务的实验管理器
├── models/spectral_prediction/    # 光谱预测模型库
│   ├── DualSpectralNet.py         # 双分支光谱网络 (CNN+Transformer)
│   └── FreqInceptionNet.py        # 频率+Inception网络 (FFT+Inception)
├── conf/                          # 模型配置文件
│   ├── dualspectralnet.yaml
│   └── freqinceptionnet.yaml
└── scripts/spectral_prediction/   # 训练脚本
    └── run_freqinceptionnet.sh
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 确保已安装PyTorch及其他核心库
pip install torch pandas numpy pyyaml scikit-learn
```

### 2. 数据准备
将您的光谱数据文件放置在 `dataset/spectral/` 目录下。

### 3. 模型训练
通过执行对应模型的shell脚本来启动训练。例如，训练 `FreqInceptionNet`：

```bash
# 首先赋予脚本执行权限
chmod +x scripts/spectral_prediction/run_freqinceptionnet.sh

# 在后台启动训练
./scripts/spectral_prediction/run_freqinceptionnet.sh
```
训练日志将保存在 `logs/` 目录下，模型检查点将保存在 `checkpoints/` 目录下。

## 🧠 支持的光谱预测模型

### 1. DualSpectralNet
- **特点**: 一个强大的双分支网络，结合了CNN和Transformer的优势。
  - **连续谱分支**: 使用CNN提取局部特征，再通过Transformer捕捉长距离依赖关系。
  - **吸收线分支**: 使用多尺度CNN（Multi-Scale CNN）来精细化提取吸收线的结构特征。
  - **融合机制**: 采用交叉注意力（Cross-Attention）来智能地融合两个分支的信息。
- **配置文件**: `conf/dualspectralnet.yaml`
- **适用场景**: 需要同时捕捉光谱的全局趋势和局部精细特征的复杂任务。

### 2. FreqInceptionNet
- **特点**: 一个创新的双分支网络，从频域和时域（空域）两个角度分析光谱。
  - **频率分支**: 将连续谱通过FFT变换到频域，再用CNN进行下采样和特征提取，专注于分析周期性和全局性特征。
  - **Inception分支**: 采用Google的Inception网络结构，通过不同大小的卷积核并行提取归一化谱的多尺度特征，并结合通道注意力（SE Block）进行特征增强。
  - **序列处理**: 在特征融合后，使用双向LSTM（BiLSTM）进一步处理序列信息。
- **配置文件**: `conf/freqinceptionnet.yaml`
- **适用场景**: 希望结合频域分析和多尺度特征提取的实验性研究。

## 🔧 自定义与扩展

### 1. 添加新模型
在 `models/spectral_prediction/` 目录下创建您的模型文件，并参考现有模型使用 `@register_model('YourModelName')` 进行注册。

### 2. 修改模型配置
直接编辑 `conf/` 目录下的YAML文件，即可调整模型的架构参数（如通道数、卷积核大小、是否使用BatchNorm等）和超参数，无需修改代码。

### 3. 调整训练脚本
`scripts/spectral_prediction/` 中的 `.sh` 脚本是训练的入口。您可以复制并修改它们，以定义不同的实验（如调整学习率、批次大小等）。

## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---
*最后更新: 2025-08-05*
