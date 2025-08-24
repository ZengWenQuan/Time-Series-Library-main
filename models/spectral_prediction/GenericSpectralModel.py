import torch
import torch.nn as nn
import yaml
from ..registries import (
    register_model,
    CONTINUUM_BRANCH_REGISTRY,
    NORMALIZED_BRANCH_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY
)

@register_model
class GenericSpectralModel(nn.Module):
    """
    通用的光谱预测主模型。
    该模型的设计是完全配置驱动的，其__init__和forward逻辑模仿了项目中的标准实现，
    使其能够灵活处理spectral_prediction和regression等不同任务。
    """
    def __init__(self, configs): # <-- 增加 sample_batch 参数
        super(GenericSpectralModel, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # --- 模仿 CustomFusionNet 的初始化逻辑 ---

        # 1. 初始化归一化谱分支 (所有任务都需要)
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(model_config['normalized_branch_config'])
        
        # 2. 根据任务类型，选择性地初始化其他模块
        if self.task_name == 'spectral_prediction':
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
            self.continuum_branch = ContBranchClass(model_config['continuum_branch_config'])
            
            FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
            fusion_config = model_config['fusion_config']
            fusion_config['dim_norm'] = self.normalized_branch.output_dim
            fusion_config['dim_cont'] = self.continuum_branch.output_dim
            self.fusion = FusionClass(fusion_config)
            head_input_dim = self.fusion.output_dim
        
        elif self.task_name == 'regression':
            head_input_dim = self.normalized_branch.output_dim
        
        else:
            raise ValueError(f"未知的任务类型: {self.task_name}")

        # 3. 初始化预测头模块
        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['head_input_dim'] = head_input_dim
        self.prediction_head = HeadClass(head_config, self.targets)

        # --- 最终版：使用传入的真实样本初始化 ---
        if model_config.get('initialize_lazy', True):
            if hasattr(configs, 'sample_batch') and configs.sample_batch is not None:
                try:
                    print("Performing forward pass with a real batch to initialize lazy modules...")
                    # 核心修改：确保传入的样本在CPU上，以匹配此时还在CPU上的模型
                    sample_batch = configs.sample_batch.to('cpu')
                    self.eval()
                    with torch.no_grad():
                        # 直接使用传入的、已经在正确设备上的样本
                        self(sample_batch)
                    self.train()
                    print("Lazy modules initialized successfully.")
                except Exception as e:
                    print(f"警告: 使用真实样本进行虚拟前向传播失败。错误: {e}")
            else:
                print("警告: 'initialize_lazy' 设置为True，但没有提供样本。跳过虚拟前向传播。")

    def forward(self, x, x_normalized=None):
        # --- 模仿 CustomFusionNet 的前向传播逻辑 ---
        if self.task_name == 'regression':
            return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            # 处理两种可能的输入格式
            if x_normalized is None and x.shape[-1] == 2:
                 x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else:
                 x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else:
            raise ValueError(f"未知的任务类型: {self.task_name}")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        return self.prediction_head(fused_sequence)

    def forward_regression(self, x):
        features = self.normalized_branch(x)
        # 将 (B, C, L) 特征直接传递给预测头，
        # 由预测头自己决定如何处理输入。
        return self.prediction_head(features)