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

## 框架深度解析 (In-Depth Framework Architecture)

本节将深入介绍框架的核心组件，包括模型库的目录结构、动态YAML配置系统以及模块的自动注册机制，旨在帮助开发者更好地理解和扩展本框架。

### 1. 模型库目录结构 (`models/`)

`models/` 目录是框架的核心，所有与模型构建相关的代码都存放于此。其子目录按功能划分，结构清晰，便于维护和扩展。

-   `models/spectral_prediction/`: **顶层模型 (Top-Level Models)**
    -   **作用**: 定义最终被执行的完整模型，例如 `DualBranchSpectralModel`。这些模型负责接收原始输入，调用不同的分支和融合模块，并最终通过预测头输出结果。
    -   **实现**: 这里的每个模型类都通过 `@register_model` 装饰器注册到主模型注册器中。

-   `models/backbones/`: **主干网络 (Backbones)**
    -   **作用**: 存放通用的、可重用的特征提取网络。这些网络通常是构成各个分支的核心，例如 `ResNet`, `InceptionTime` 等。
    -   **实现**: 模块通过 `@register_backbone` 装饰器注册。

-   `models/submodules/`: **基础子模块 (Submodules)**
    -   **作用**: 存放构成更复杂模块的基础组件，是模型架构中最细粒度的可重用单元。例如，`attention.py` 中的各种注意力实现、`fusion_blocks.py` 中的特征融合块等。
    -   **实现**: 根据功能，使用 `@register_block`, `@register_fusion` 等装饰器注册。

-   `models/heads/`: **预测头 (Prediction Heads)**
    -   **作用**: 模型的最后一部分，负责将经过主干网络和融合模块处理后的高级特征映射到最终的输出（例如，恒星参数）。
    -   **实现**: 模块通过 `@register_head` 装饰器注册。

-   `models/blocks/`: **通用模块块 (General Blocks)**
    -   **作用**: 存放比 `submodules` 更大一些的通用模块，这些模块本身可能由多个子模块构成，但又不足以成为一个完整的分支或主干。
    -   **实现**: 模块通过 `@register_block` 装饰器注册。

-   `models/registries.py`: **注册器中心 (Registry Hub)**
    -   **作用**: 定义了项目中所有的注册器实例（如 `MODEL_REGISTRY`, `BACKBONES`）和注册装饰器（如 `@register_model`, `@register_backbone`）。这是实现框架“即插即用”特性的基石。

### 2. 动态YAML配置系统

本框架采用了一套强大且灵活的动态配置系统，允许用户通过组合不同的YAML文件来构建复杂的模型，而无需修改任何Python代码。

**核心机制**: 该系统的核心逻辑位于 `exp/exp_basic.py` 的 `_load_and_merge_configs` 方法中。其工作流程如下：

1.  **加载主配置文件**: 训练开始时，程序会首先加载一个**主配置文件**（例如 `conf/dualbranchspectral.yaml`）。
2.  **解析模块名称**: 主配置文件中并不包含所有参数，而是为模型的各个部分（如 `backbone_name`, `fusion_name`）指定一个名称。
3.  **动态加载子配置**: 程序根据主配置中指定的名称，在 `conf/` 下对应的子目录中查找并加载**子配置文件**。例如，如果主配置中指定 `backbone_name: "ResNet50Backbone1D"`，程序就会自动加载 `conf/backbone/ResNet50Backbone1D.yaml` 文件。
4.  **合并配置**: 所有子配置文件的内容会被合并到主配置中，形成一个完整的配置字典，供模型初始化使用。

**示例**:

假设我们有以下文件结构：
```
conf/
├── dualbranchspectral.yaml   # 主配置
└── backbone/
    └── ResNet50Backbone1D.yaml # 子配置
```

**主配置 (`dualbranchspectral.yaml`)**:
```yaml
name: "DualBranchSpectralModel"
backbone_name: "ResNet50Backbone1D"  # <-- 指向子配置的名称
# ... 其他分支和头的名称 ...
```

**子配置 (`backbone/ResNet50Backbone1D.yaml`)**:
```yaml
# ResNet50Backbone1D 模块的具体参数
in_channels: 1
layers: [3, 4, 6, 3]
# ... 其他参数 ...
```

在运行时，框架会自动将 `ResNet50Backbone1D.yaml` 的内容加载到最终配置的 `backbone_config` 键下，模型在初始化时便可直接访问这些详细参数。

### 3. 模块自动注册机制

为了实现上述配置系统的动态加载，框架使用了一套基于装饰器的自动注册机制。

**核心机制**: 该机制的核心位于 `models/registries.py` 文件中。

1.  **注册器实例**: 文件中定义了多个全局字典作为注册器，例如 `MODEL_REGISTRY` 用于存放顶层模型，`BACKBONES` 用于存放主干网络等。
2.  **注册装饰器**: 文件为每种类型的模块提供了对应的注册装饰器，例如 `@register_model`, `@register_backbone`。
3.  **自动注册**: 当你在代码中定义一个新模块并为其加上相应的装饰器时，该模块的类名（或自定义名称）和类本身会自动被添加到一个全局注册器字典中。

**示例**: 如何注册一个新的主干网络。

```python
# in models/backbones/my_new_backbone.py

from torch import nn
from models.registries import register_backbone # 1. 导入装饰器

@register_backbone # 2. 使用装饰器注册
class MyNewBackbone(nn.Module):
    def __init__(self, config): # 构造函数接收合并后的配置字典
        super().__init__()
        # ... 你的网络实现 ...

    def forward(self, x):
        # ... 前向传播 ...
        return x
```

**工作流程**:
1.  当Python解释器加载 `my_new_backbone.py` 文件时，`@register_backbone` 装饰器会立即执行。
2.  它将 `MyNewBackbone` 类的名称 `"MyNewBackbone"` 作为键，类本身作为值，存入 `BACKBONES` 这个全局字典中。
3.  在模型构建时 (`_build_model` 函数)，代码从YAML配置中读取模型/模块的名称（例如 `"MyNewBackbone"`）。
4.  然后以该名称为键，在对应的注册器字典（例如 `BACKBONES`）中查找并获取到 `MyNewBackbone` 这个类。
5.  最后，使用获取到的类和配置字典来实例化模块：`MyNewBackbone(config)`。

通过这套机制，任何新添加的、只要被正确注册的模块，都可以立即通过修改YAML配置文件来被框架发现和使用，实现了真正意义上的“即插即用”。


## 📄 许可证

本项目遵循MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

---

*最后更新: 2025-08-24*