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
        self.model_config = model_config # Save config for profiling

        # --- 新增：获取全局设置并向下传递 ---
        global_settings = model_config.get('global_settings', {})

        # --- 模仿 CustomFusionNet 的初始化逻辑 ---

        # 1. 初始化归一化谱分支 (所有任务都需要)
        norm_branch_config = model_config['normalized_branch_config']
        norm_branch_config.update(global_settings)
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(norm_branch_config)
        
        # 2. 初始化所有分支和融合模块
        cont_branch_config = model_config['continuum_branch_config']
        cont_branch_config.update(global_settings)
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
        self.continuum_branch = ContBranchClass(cont_branch_config)
        
        fusion_config = model_config['fusion_config']
        fusion_config.update(global_settings)
        FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
        fusion_config['channels_norm'] = self.normalized_branch.output_channels
        fusion_config['channels_cont'] = self.continuum_branch.output_channels
        fusion_config['len_norm'] = self.normalized_branch.output_length
        fusion_config['len_cont'] = self.continuum_branch.output_length
        self.fusion = FusionClass(fusion_config)
        head_input_channels = self.fusion.output_channels

        # 3. 初始化预测头模块
        head_config = model_config['head_config']
        head_config.update(global_settings)
        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config['head_input_channels'] = head_input_channels
        head_config['head_input_length'] = self.fusion.output_length
        head_config['targets'] = self.targets
        self.prediction_head = HeadClass(head_config)


    def profile_model(self, sample_batch):
        """
        Calculates and returns the FLOPs and parameters for each submodule.
        """
        from thop import profile
        stats = {}
        device = sample_batch.device
        
        # --- 模仿 forward pass 来获取各部分的输入 ---
        if self.task_name == 'regression':
            x_continuum, x_normalized = sample_batch, sample_batch

        elif self.task_name == 'spectral_prediction':
            x=sample_batch
            if x.dim() == 3 and x.shape[-1] == 2:
                 x_continuum, x_normalized = x[:, :, 0].unsqueeze(1), x[:, :, 1].unsqueeze(1)
            elif x.dim()==2 and x.shape[-1]==2:
                x_continuum, x_normalized = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)
        
        # 对两种任务统一进行性能分析
        # 1. Normalized Branch
        macs, params = profile(self.normalized_branch, inputs=(x_normalized,), verbose=False)
        stats['normalized_branch'] = {'flops': macs * 2, 'params': params}
        features_norm = self.normalized_branch(x_normalized)

        # 2. Continuum Branch
        macs, params = profile(self.continuum_branch, inputs=(x_continuum,), verbose=False)
        stats['continuum_branch'] = {'flops': macs * 2, 'params': params}
        features_cont = self.continuum_branch(x_continuum)

        # 3. Fusion Layer
        macs, params = profile(self.fusion, inputs=(features_norm, features_cont), verbose=False)
        stats['fusion'] = {'flops': macs * 2, 'params': params}
        fused_sequence = self.fusion(features_norm, features_cont)

        # 4. Prediction Head
        macs, params = profile(self.prediction_head, inputs=(fused_sequence,), verbose=False)
        stats['prediction_head'] = {'flops': macs * 2, 'params': params}
        
        return stats
                

    def forward(self, x, x_normalized=None):
        # --- 模仿 CustomFusionNet 的前向传播逻辑 ---
        if self.task_name == 'regression':
            return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            # 处理两种可能的输入格式
            if x.dim() == 3 and x.shape[-1] == 2:
                 x_continuum, x_normalized = x[:, :, 0].unsqueeze(1), x[:, :, 1].unsqueeze(1)
            elif x.dim()==2 and x.shape[-1]==2:
                x_continuum, x_normalized = x[:, 0].unsqueeze(1), x[:, 1].unsqueeze(1)

            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else:
            raise ValueError(f"未知的任务类型: {self.task_name}")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        return self.prediction_head(fused_sequence)

    def forward_regression(self, x):
        # 将单个输入序列传递给两个分支
        features_norm = self.normalized_branch(x)
        features_cont = self.continuum_branch(x)
        
        # 融合特征并进行预测
        fused_sequence = self.fusion(features_norm, features_cont)
        return self.prediction_head(fused_sequence)

    