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
class DualPyramidNet(nn.Module):
    def __init__(self, configs):
        super(DualPyramidNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        # Branches
        BranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['branch_name']]
        self.normalized_extractor = BranchClass(model_config['branch_config'])
        if self.task_name == 'spectral_prediction':
            self.continuum_extractor = BranchClass(model_config['branch_config'])
            
            # Fusion
            FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
            fusion_config = model_config['fusion_config']
            fusion_config['dim_norm'] = self.normalized_extractor.output_dim
            fusion_config['dim_cont'] = self.continuum_extractor.output_dim
            self.fusion = FusionClass(fusion_config)
            head_input_dim = self.fusion.output_dim
        else: # regression
            head_input_dim = self.normalized_extractor.output_dim

        # Head
        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['ffn_input_dim'] = head_input_dim
        self.prediction_head = HeadClass(head_config, self.targets, self.label_size)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': 
            return self.prediction_head(self.normalized_extractor(x))
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: 
                x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: 
                x_continuum = x
            continuum_features = self.continuum_extractor(x_continuum)
            normalized_features = self.normalized_extractor(x_normalized)
            fused_features = self.fusion(normalized_features, continuum_features)
            return self.prediction_head(fused_features)