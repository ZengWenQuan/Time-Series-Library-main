
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

from ..submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from ..submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY
from ..submodules.fusion_heads import HEAD_REGISTRY, FUSION_REGISTRY

@register_model('LargeKernelConvNet')
class LargeKernelConvNet(nn.Module):
    def __init__(self, configs):
        super(LargeKernelConvNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        # Branches
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.upsample_branch = NormBranchClass(model_config['upsample_branch_config'])
        
        if self.task_name == 'spectral_prediction':
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
            self.large_kernel_branch = ContBranchClass(model_config['large_kernel_branch_config'])
            
            # Fusion
            FusionClass = FUSION_REGISTRY[model_config['fusion_name']]
            fusion_config = model_config['fusion_config']
            fusion_config['dim_norm'] = self.upsample_branch.output_dim
            fusion_config['dim_cont'] = self.large_kernel_branch.output_dim
            self.fusion = FusionClass(fusion_config)
            head_input_dim = self.fusion.output_dim
        else: # regression
            head_input_dim = self.upsample_branch.output_dim

        # Head
        HeadClass = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['head_input_dim'] = head_input_dim
        self.prediction_head = HeadClass(head_config, self.targets)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': 
            return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: 
                x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: 
                x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_cont = self.large_kernel_branch(x_continuum)
        features_norm = self.upsample_branch(x_normalized)
        fused_features = self.fusion(features_norm, features_cont)
        return self.prediction_head(fused_features)

    def forward_regression(self, x):
        features_map = self.upsample_branch(x)
        features_seq = features_map.transpose(1, 2)
        return self.prediction_head(features_seq)
