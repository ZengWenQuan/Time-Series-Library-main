
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..registries import (
    register_model,
    CONTINUUM_BRANCH_REGISTRY,
    NORMALIZED_BRANCH_REGISTRY,
    FUSION_REGISTRY,
    HEAD_REGISTRY
)

@register_model
class DualBranchMoENet(nn.Module):
    def __init__(self, configs):
        super(DualBranchMoENet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: config = yaml.safe_load(f)
        
        global_settings = config.get('global_settings', {})
        norm_type = config['normalization_type']

        continuum_branch_config = config['continuum_branch_config']
        continuum_branch_config.update(global_settings)
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[config['continuum_branch_name']]
        self.freq_branch = ContBranchClass(continuum_branch_config, norm_type)

        normalized_branch_config = config['normalized_branch_config']['pyramid_with_attention']
        normalized_branch_config.update(global_settings)
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[config['normalized_branch_name']]
        self.line_branch = NormBranchClass(normalized_branch_config, norm_type)

        if self.task_name == 'spectral_prediction':
            fusion_config = config['fusion_config']
            fusion_config.update(global_settings)
            FusionClass = FUSION_REGISTRY[config['fusion_name']]
            fusion_config['channels_cont'] = self.freq_branch.output_channels
            fusion_config['channels_norm'] = self.line_branch.output_channels
            self.fusion = FusionClass(fusion_config)
            head_input_dim = self.fusion.output_dim
        else: # regression
            head_input_dim = self.line_branch.output_dim

        head_config = config['head_config']
        head_config.update(global_settings)
        HeadClass = HEAD_REGISTRY[config['head_name']]
        head_config['head_input_dim'] = head_input_dim
        head_config['targets'] = self.targets
        self.prediction_head = HeadClass(head_config)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1: x_continuum = x_continuum.squeeze(-1)
        freq_features = self.freq_branch(x_continuum)
        line_features = self.line_branch(x_normalized)
        
        # The fusion module is now responsible for handling different lengths if necessary
        fused_features = self.fusion(line_features, freq_features)
        return self.prediction_head(fused_features)

    def forward_regression(self, x):
        features = self.line_branch(x).permute(0, 2, 1)
        return self.prediction_head(features)
