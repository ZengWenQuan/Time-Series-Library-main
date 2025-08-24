
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from ..registries import (
    CONTINUUM_BRANCH_REGISTRY,
    NORMALIZED_BRANCH_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY,
    register_model
)

@register_model
class CustomFusionNet(nn.Module):
    def __init__(self, configs):
        super(CustomFusionNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(model_config['normalized_branch_config'])
        
        if self.task_name == 'spectral_prediction':
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
            cont_branch_config = model_config['continuum_branch_config']
            cont_branch_config['feature_size'] = configs.feature_size
            self.continuum_branch = ContBranchClass(cont_branch_config)
            
            FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
            fusion_config = model_config['fusion_config']
            fusion_config['dim_norm'] = self.normalized_branch.output_dim
            fusion_config['dim_cont'] = self.continuum_branch.output_dim
            self.fusion = FusionClass(fusion_config)
            lstm_input_dim = self.fusion.output_dim
        else: # regression
            lstm_input_dim = self.normalized_branch.output_dim

        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['lstm_input_dim'] = lstm_input_dim
        self.prediction_head = HeadClass(head_config, self.targets)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        return self.prediction_head(fused_sequence)

    def forward_regression(self, x):
        features = self.normalized_branch(x)
        features = features.transpose(1, 2)
        return self.prediction_head(features)
