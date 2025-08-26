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
class DualSpectralNet(nn.Module):
    def __init__(self, configs):
        super(DualSpectralNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        self.feature_size = configs.feature_size

        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        global_settings = model_config.get('global_settings', {})

        # Instantiate branches
        continuum_branch_config = model_config['continuum_branch_config']
        continuum_branch_config.update(global_settings)
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
        self.continuum_branch = ContBranchClass(continuum_branch_config)

        normalized_branch_config = model_config['normalized_branch_config']
        normalized_branch_config.update(global_settings)
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(normalized_branch_config)

        # Instantiate fusion module
        fusion_config = model_config['fusion_config']
        fusion_config.update(global_settings)
        FusionModule = FUSION_REGISTRY[model_config['fusion_name']]
        fusion_config['channels_cont'] = self.continuum_branch.output_channels
        fusion_config['channels_norm'] = self.normalized_branch.output_channels
        self.fusion = FusionModule(fusion_config)

        # Instantiate prediction head
        head_config = model_config['head_config']
        head_config.update(global_settings)
        PredictionHead = HEAD_REGISTRY[model_config['head_name']]
        head_config['head_input_dim'] = self.fusion.output_dim
        head_config['targets'] = self.targets
        self.prediction_head = PredictionHead(head_config)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'spectral_prediction':
            if x_normalized is None:
                x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else:
                x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else:
            # Handle other tasks if necessary
            pass

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        continuum_features = self.continuum_branch(x_continuum)
        normalized_features = self.normalized_branch(x_normalized)
        
        fused_features = self.fusion(normalized_features, continuum_features)
        
        predictions = self.prediction_head(fused_features)
        
        return predictions
