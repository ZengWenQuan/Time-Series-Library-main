# 恒星光谱参数预测框架

## 核心设计思想：模块化与可扩展性

本项目是一个专注于恒星光谱参数预测的深度学习框架，其核心设计思想是将一个完整的光谱分析模型解耦为三个可独立替换的关键部分：

1.  **分支 (Branch)**: 负责处理特定类型的输入（如连续谱或归一化谱）。项目中内置了多种分支实现，例如基于CNN、小波变换、混合专家（MoE）或大卷积核等。
2.  **融合模块 (Fusion)**: 负责将来自不同分支的特征进行智能融合。支持包括简单拼接（Concatenation）、逐元素相加（Addition）、交叉注意力（Cross-Attention）等多种策略。
3.  **预测头 (Head)**: 接收融合后的特征，并执行最终的回归或分类任务。

这种“即插即用”的设计允许研究人员通过简单的修改YAML配置文件，就能轻松地组合、测试和创造出新的模型架构，而无需深入修改底层代码。所有的模块都通过注册器模式（Registry Pattern）进行管理，保证了框架的灵活性和易于扩展性。

## 🗂️ 项目结构

```
├── data_provider/    # 数据加载器
├── dataset/split_data/ # 训练/验证/测试数据存放目录
├── exp/              # 实验管理器
├── models/           # 模型库 (包含分支、融合、头等子模块)
├── conf/             # 模型及统计数据配置文件
├── scripts/          # 训练脚本
└── update_stats.py   # 更新统计数据的脚本
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保已安装PyTorch及其他核心库
pip install torch pandas numpy pyyaml scikit-learn
```

### 2. 数据准备

将您的光谱数据文件放置在 `dataset/` 目录下，并划分为 `train`, `val`, `test` 子目录。

## ‼️ 关键步骤：生成统计数据文件 (stats.yaml)

**警告：这是成功训练模型的关键一步，直接关系到预测结果的准确性。**

本框架使用 `conf/stats.yaml` 文件来对所有输入数据和输出标签进行归一化。归一化所用的统计数据（如均值、标准差）**必须只从训练集中计算**，以避免数据泄露。

我们提供了一个专门的脚本 `update_stats.py` 来正确地生成此文件。

**使用方法:**

在开始任何训练之前，请先运行以下命令：

```bash
python update_stats.py
```

该命令会使用默认路径 `dataset/split_data/train` 的数据来更新 `conf/stats.yaml` 文件。正确生成 `stats.yaml` 后，您才可以开始下一步的模型训练。

### 3. 模型训练

通过执行对应模型的shell脚本来启动训练。例如:

```bash
chmod +x scripts/spectral_prediction/run_flexiblefusionnet.sh
./scripts/spectral_prediction/run_flexiblefusionnet.sh
```

## 🧠 核心模型库

以下是项目中预置的几个核心模型，它们展示了框架的不同组合方式。

### 1. FlexibleFusionNet

- **设计思想**: 一个灵活的“混合专家”模型，旨在让光谱的不同特征被最擅长处理它的“专家”网络所分析。
- **连续谱分支**: `CustomMoEBranch` - 使用混合专家网络处理FFT变换后的频域特征。
- **归一化谱分支**: `MultiScalePyramidBranch` - 一个标准的多尺度金字塔CNN，用于提取谱线特征。
- **融合策略**: `add` - 将两个分支的特征逐元素相加。
- **配置文件**: `conf/flexiblefusionnet.yaml`
- **训练脚本**: `scripts/spectral_prediction/run_flexiblefusionnet.sh`

### 2. CustomFusionNet

- **设计思想**: 一个高度可定制的双分支融合网络，是快速实验新想法的理想基础。
- **连续谱分支**: `ContinuumWaveletBranch` - 采用小波变换提取时频特征。
- **归一化谱分支**: `NormalizedSpectrumBranch` - 采用金字塔式多尺度网络捕捉精细特征。
- **融合策略**: `concat` - 将两个分支的特征拼接在一起。
- **配置文件**: `conf/customfusionnet.yaml`
- **训练脚本**: `scripts/spectral_prediction/run_customfusionnet.sh`

### 3. LargeKernelConvNet

- **设计思想**: 探索使用超大卷积核来捕捉光谱长程依赖的可能性。
- **连续谱分支**: `LargeKernelBranch` - 使用单个超大核卷积层直接处理整个光谱，输出为向量。
- **归一化谱分支**: `UpsampleMultiScaleBranch` - 先通过转置卷积提升分辨率，再用多尺度网络提取特征。
- **融合策略**: `ConcatFusion` - 将向量分支的输出广播后与序列分支的特征进行拼接。
- **配置文件**: `conf/largekernel.yaml`
- **训练脚本**: `scripts/spectral_prediction/run_largekernel.sh`

## 🔧 自定义与扩展

本框架的核心优势在于其高度的可扩展性。你可以通过以下两种方式来构建、测试和验证你的新想法：

### 1. 修改现有模型（无需编写代码）

最简单的方式是修改 `conf/` 目录下的YAML配置文件。每个模型都由 `branch_continuum`、`branch_normalized`、`fusion` 和 `head` 四个核心部分组成。

例如，要修改 `CustomFusionNet`，你可以打开 `conf/customfusionnet.yaml` 文件：

```yaml
# conf/customfusionnet.yaml
model:
  name: "CustomFusionNet"
  branch_continuum:
    name: "ContinuumWaveletBranch"  # <-- 更换连续谱分支
    # ...
  branch_normalized:
    name: "NormalizedSpectrumBranch" # <-- 更换归一化谱分支
    # ...
  fusion:
    name: "concat" # <-- 更换融合策略
    # ...
  head:
    name: "DecoderHead" # <-- 更换预测头
    # ...
```

你只需要将 `name` 字段的值更改为已注册的任何其他模块名称，即可轻松地“即插即用”，组合出全新的模型架构。所有可用的模块都定义在 `models/` 目录下。

### 2. 添加新模块（需要编写代码）

如果你有全新的模块设计，可以按照以下三个步骤将其集成到框架中：

**步骤一：创建模块文件**
在对应的子目录下创建一个新的Python文件。例如，要创建一个新的归一化谱分支：
`models/spectral_prediction/branch/MyNewBranch.py`

**步骤二：实现模块**
在新文件中，编写你的PyTorch模块。

```python
# models/spectral_prediction/branch/MyNewBranch.py
from torch import nn
from models.registries import NORMALIZED_BRANCH_REGISTRY

@NORMALIZED_BRANCH_REGISTRY.register()
class MyNewBranch(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # ... 你的网络层定义 ...

    def forward(self, x):
        # ... 你的前向传播逻辑 ...
        return x
```

**步骤三：注册模块**
使用对应的注册器装饰器（Decorator）来注册你的新模块。关键在于 `@NORMALIZED_BRANCH_REGISTRY.register()` 这一行，它会自动将 `MyNewBranch` 添加到可用模块列表中。

- **连续谱分支**: `@CONTINUUM_BRANCH_REGISTRY.register()`
- **归一化谱分支**: `@NORMALIZED_BRANCH_REGISTRY.register()`
- **融合模块**: `@FUSION_REGISTRY.register()`
- **预测头**: `@HEAD_REGISTRY.register()`

完成这三步后，你就可以在YAML配置文件中使用 `"MyNewBranch"` 了。

### 3. 调整训练过程

模型的训练过程同样通过YAML文件进行配置。在每个模型的配置文件中，都有一个 `training` 部分，你可以在这里调整超参数。

```yaml
# conf/customfusionnet.yaml
training:
  loss_function: "mse"       # 损失函数 (例如: "mse", "mae")
  optimizer: "adam"          # 优化器
  learning_rate: 0.001       # 学习率
  weight_decay: 0.0001       # 权重衰减
  scheduler: "cosine"        # 学习率调度器
  epochs: 100                # 训练轮次
  batch_size: 64             # 批处理大小
```

通过修改这些参数，你可以精细地控制模型的训练行为，以适应不同的实验需求。


## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

*最后更新: 2025-08-24*
