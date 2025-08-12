
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
from exp.exp_basic import register_model
from models.submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from models.submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY

class CrossAttentionFusion(nn.Module):
    def __init__(self, continuum_dim, absorption_dim, fusion_dim, num_heads=4, dropout_rate=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.continuum_proj = nn.Linear(continuum_dim, fusion_dim)
        self.absorption_proj = nn.Linear(absorption_dim, fusion_dim)
        self.norm_cont = nn.LayerNorm(fusion_dim)
        self.norm_abs = nn.LayerNorm(fusion_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        
    def forward(self, continuum_features, absorption_features):
        cont_proj = self.continuum_proj(continuum_features).unsqueeze(1)
        abs_proj = self.absorption_proj(absorption_features).unsqueeze(1)
        norm_cont_proj = self.norm_cont(cont_proj)
        norm_abs_proj = self.norm_abs(abs_proj)
        cont_attended, _ = self.cross_attention(norm_cont_proj, norm_abs_proj, norm_abs_proj)
        abs_attended, _ = self.cross_attention(norm_abs_proj, norm_cont_proj, norm_cont_proj)
        fused_features = torch.cat([cont_attended.squeeze(1), abs_attended.squeeze(1)], dim=1)
        return self.fusion_layer(fused_features)

@register_model('DualSpectralNet')
class DualSpectralNet(nn.Module):
    def __init__(self, configs):
        super(DualSpectralNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
        self.continuum_branch = ContBranchClass(model_config['continuum_branch_config'])

        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.absorption_branch = NormBranchClass(model_config['normalized_branch_config'])
        
        fusion_conf = model_config['fusion_config']
        self.feature_fusion = CrossAttentionFusion(
            continuum_dim=self.continuum_branch.output_dim,
            absorption_dim=self.absorption_branch.output_dim,
            fusion_dim=fusion_conf['hidden_dim'],
            num_heads=fusion_conf['num_heads'],
            dropout_rate=fusion_conf.get('dropout', 0.1))
        
        self.prediction_head = self._build_prediction_head(fusion_conf['hidden_dim'], model_config['prediction_head_config'])
        self._initialize_weights()

    def _build_prediction_head(self, input_dim, config):
        layers, current_dim = [], input_dim
        for hidden_dim in config['hidden_dims']:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(config.get('dropout', 0.1))])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.label_size))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)): init.uniform_(m.weight, a=-0.1, b=0.1)
            if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None: init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None: init.constant_(m.bias, 0)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :self.feature_size], x[:, self.feature_size:]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        continuum_features = self.continuum_branch(x_continuum)
        absorption_features = self.absorption_branch(x_normalized)
        fused_features = self.feature_fusion(continuum_features, absorption_features)
        return self.prediction_head(fused_features)

    def forward_regression(self, x):
        # For regression, we assume the normalized branch (absorption branch) is the primary feature extractor
        features = self.absorption_branch(x)
        return self.prediction_head(features)
