
# 如何使用该光谱预测框架

本文档提供了在本框架中进行恒星光谱参数预测任务的详细操作指南，包括环境设置、数据准备、模型训练、实验跟踪和高级自定义。

---

## 1. 安装与环境设置

建议使用 `conda` 来管理项目环境，以确保依赖隔离和一致性。

```bash
# 1. 克隆项目 (如果尚未操作)
# git clone <your-repo-url>
# cd Time-Series-Library-main

# 2. 创建并激活Conda环境 (推荐使用Python 3.9+)
conda create -n lamost python=3.10
conda activate lamost

# 3. 安装核心依赖
# PyTorch (请根据您的CUDA版本访问PyTorch官网获取最合适的命令)
# pip install torch torchvision torchaudio

# 4. 安装其他必要的库
pip install pandas numpy pyyaml scikit-learn matplotlib

# 5. 安装实验跟踪工具MLflow
pip install mlflow
```

## 2. 项目结构概览

了解关键目录有助于您快速定位和修改代码：

- `conf/`: 存放所有模型的 `.yaml` 配置文件。
- `data_provider/`: 包含数据加载和预处理的逻辑，核心是 `data_loader_spectral.py`。
- `dataset/spectral/`: **您的光谱数据文件应放置在此处**。
- `exp/`: 包含实验的核心训练、验证和测试逻辑，核心是 `exp_spectral_prediction.py`。
- `models/spectral_prediction/`: 存放所有光谱预测模型的Python实现。
- `scripts/spectral_prediction/`: **存放用于启动不同模型训练的 `.sh` 脚本**。
- `runs/`: 保存每个实验的输出，包括日志、图表和模型检查点。
- `mlruns/`: **MLflow的实验数据存储目录**，由MLflow自动创建和管理。

## 3. 数据准备

请确保您的数据文件已按以下结构放置在 `dataset/spectral/` 目录下：

- `final_spectra_continuum.csv`: 包含连续谱特征数据。
- `final_spectra_normalized.csv`: 包含归一化谱特征数据。
- `removed_with_rv.csv`: 包含目标参数标签（Teff, logg, FeH, CFe）。

*注意：数据加载器会根据 `obsid` 列来对齐这三个文件中的样本。*

## 4. 运行一个实验（核心工作流）

#### **步骤 4.1: 选择并配置训练脚本**

进入 `scripts/spectral_prediction/` 目录，选择一个您想运行的脚本，例如 `run_freqinception_ln.sh`。

您可以直接编辑此文件，以修改传递给 `run.py` 的参数，例如：
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--train_epochs`: 训练轮数

#### **步骤 4.2: 赋予脚本执行权限**

这是一个必需的、一次性的步骤。

```bash
chmod +x scripts/spectral_prediction/run_freqinception_ln.sh
```

#### **步骤 4.3: 运行脚本**

执行脚本以在前台启动训练。所有日志将直接打印到您的终端。

```bash
./scripts/spectral_prediction/run_freqinception_ln.sh
```

## 5. 使用MLflow跟踪和分析实验

MLflow为您提供了一个强大的本地Web界面来查看和比较所有实验。

### 场景一：在本地机器上运行

#### **步骤 5.1: 启动MLflow UI**

训练开始后，**打开一个新的终端窗口**，进入项目根目录，然后运行：

```bash
mlflow ui
```

*提示：这会启动一个本地Web服务器，它会持续运行。您可以让这个终端一直开着。*

#### **步骤 5.2: 在浏览器中访问**

打开您的Web浏览器，访问地址：[http://127.0.0.1:5000](http://127.0.0.1:5000)

### 场景二：在远程服务器上运行

当您在服务器上运行 `mlflow ui` 时，它默认只监听服务器的内部地址(`127.0.0.1`)，您无法直接从您自己的电脑上访问。正确的解决方案是使用 **SSH端口转发**，它安全、高效。

#### **步骤 5.2.1: 建立SSH端口转发连接**

在**您自己的本地电脑上**，打开一个新终端，使用以下命令登录到您的服务器。请将 `your_username` 和 `your_server_ip` 替换为您的实际凭据。

```bash
ssh -L 5000:127.0.0.1:5000 your_username@your_server_ip
```

- **命令解释**: `-L` 参数创建了一个安全的“管道”。它将发送到您**本地电脑** `5000` 端口的请求，安全地转发到**服务器**的 `5000` 端口。
- **重要**: 请保持这个SSH连接窗口不要关闭，因为它是您的安全管道。

#### **步骤 5.2.2: 在服务器上启动MLflow UI**

在刚刚建立连接的SSH窗口中，进入项目目录并启动MLflow服务：

```bash
cd /path/to/your/project  # 切换到您的项目目录
mlflow ui
```

#### **步骤 5.2.3: 在本地浏览器中访问**

现在，回到**您自己本地电脑的浏览器**，像在本地一样访问以下地址：

[http://127.0.0.1:5000](http://127.0.0.1:5000)

您现在看到的，就是服务器上MLflow的实时界面。

### 分析结果

无论使用哪种场景，在MLflow界面中，您都可以：
1.  在左侧面板看到名为 `spectral_prediction` 的实验。
2.  点击它，主界面会显示该实验下的所有“运行(Runs)”，每次执行训练脚本都会产生一次新的运行。
3.  点击任何一次运行，您可以查看：
    - **Parameters**: 本次运行使用的所有超参数。
    - **Metrics**: 训练和验证损失、学习率等指标随时间变化的交互式图表。
    - **Artifacts**: 本次运行保存的所有产物，包括 `loss_curve.pdf`, `lr_curve.pdf`，以及最佳模型检查点。

## 6. 高级用法：自定义与扩展

#### **创建新实验**

最简单的方法是复制一个现有的训练脚本：

1.  `cp scripts/spectral_prediction/run_freqinception_ln.sh scripts/spectral_prediction/my_new_experiment.sh`
2.  编辑 `my_new_experiment.sh`，修改 `--model_id` 以便区分，并调整您想测试的参数（如 `--learning_rate`）。
3.  运行新脚本。一次新的运行将自动出现在MLflow UI中。

#### **创建新模型**

1.  在 `models/spectral_prediction/` 目录下创建一个新的模型文件，例如 `MyAwesomeNet.py`。
2.  参考 `FreqInceptionLNet.py` 的结构实现您的模型，并使用 `@register_model('MyAwesomeNet')` 装饰器进行注册。
3.  在 `conf/` 目录下为您的新模型创建一个配置文件 `myawesomenet.yaml`。
4.  创建一个新的训练脚本，将其中的 `--model` 参数设为 `MyAwesomeNet`，并将 `--model_conf` 指向您的新配置文件。

## 7. 常见问题与排查 (Troubleshooting)

- **问题: 训练一开始，损失就变成 `NaN`？**
  - **原因 1 (最常见)**: 数值不稳定。即使数据已经标准化，卷积层或FFT等操作仍可能在混合精度(FP16)或处理极端离群点时产生 `NaN`。
  - **解决方案**: 
    a. 尝试使用 `LayerNorm` 替代 `BatchNorm` (参考 `FreqInceptionLNet` 模型)。
    b. 检查您的数据加载器，确保对输入数据进行了裁剪(`np.clip`)或更稳健的缩放。
    c. 确保训练脚本中启用了梯度裁剪 (`--gradient_clip_val`)。

- **问题: 验证损失曲线非常不稳定，上下剧烈震荡？**
  - **原因**: 这是 `BatchNorm` 在小批次训练下的典型副作用。训练时和验证时的统计数据不一致。
  - **解决方案**: 
    a. 换用 `LayerNorm` (参考 `FreqInceptionLNet` 模型)。
    b. 在 `BatchNorm1d` 层中设置一个更小的 `momentum` 值，例如 `momentum=0.01`。
    c. 在显存允许的情况下，增大训练的批次大小。

---
*本文档旨在提供清晰的操作流程，祝您实验顺利！*
