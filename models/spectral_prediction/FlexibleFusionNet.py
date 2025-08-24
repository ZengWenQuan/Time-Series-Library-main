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
class FlexibleFusionNet(nn.Module):
    """
    灵活融合网络 (FlexibleFusionNet)。
    
    该模型集成了我们新设计的以下模块：
    1. CustomMoEBranch: 处理连续谱，门控在频域，专家在时域。
    2. MultiScalePyramidBranch: 处理归一化谱，使用多尺度卷积。
    3. MultiTaskHead: 为每个目标使用独立的预测头。
    4. GeneralFusion: 可配置的特征融合模块。
    """
    def __init__(self, configs):
        super(FlexibleFusionNet, self).__init__()
        self.task_name = configs.task_name
        # 统一从configs对象获取targets
        self.targets = configs.targets
        
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # 1. 从注册表查找并初始化分支
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
        self.continuum_branch = ContBranchClass(model_config['continuum_branch_config'])
        
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(model_config['normalized_branch_config'])

        # 2. 从注册表查找并初始化融合模块
        FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
        fusion_config = model_config['fusion_config']
        fusion_config['dim_cont'] = self.continuum_branch.output_dim
        fusion_config['dim_norm'] = self.normalized_branch.output_dim
        self.fusion = FusionClass(fusion_config)

        # 3. 从注册表查找并初始化多任务预测头
        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['head_input_dim'] = self.fusion.output_dim
        self.prediction_head = HeadClass(head_config, self.targets)

    def forward(self, x_continuum, x_normalized, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播
        """
        # 分别通过两个分支提取特征
        cont_features = self.continuum_branch(x_continuum)
        norm_features = self.normalized_branch(x_normalized)

        # 融合两个分支的特征
        # 注意：融合模块内部会将输入压平为 (B, C*L) 的二维张量
        fused_features = self.fusion(norm_features, cont_features)

        # 通过多任务预测头得到最终预测结果
        predictions = self.prediction_head(fused_features)

        return predictions