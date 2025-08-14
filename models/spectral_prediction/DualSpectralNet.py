import torch
import torch.nn as nn
import yaml
from exp.exp_basic import register_model
from ..submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from ..submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY
from ..submodules.fusion_heads import FUSION_REGISTRY, HEAD_REGISTRY

@register_model('DualSpectralNet')
class DualSpectralNet(nn.Module):
    def __init__(self, configs):
        super(DualSpectralNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        self.label_size = len(self.targets)
        self.feature_size = configs.feature_size

        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # Instantiate branches
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
        self.continuum_branch = ContBranchClass(model_config['continuum_branch_config'])

        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(model_config['normalized_branch_config'])

        # Instantiate fusion module
        FusionModule = FUSION_REGISTRY[model_config['fusion_name']]
        fusion_config = model_config['fusion_config']
        fusion_config['dim_cont'] = self.continuum_branch.output_dim
        fusion_config['dim_norm'] = self.normalized_branch.output_dim
        self.fusion = FusionModule(fusion_config)

        # Instantiate prediction head
        PredictionHead = HEAD_REGISTRY[model_config['head_name']]
        head_config = model_config['head_config']
        head_config['input_dim'] = self.fusion.output_dim
        self.prediction_head = PredictionHead(head_config, self.label_size)

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
