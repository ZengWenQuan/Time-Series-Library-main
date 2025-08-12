import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model
import torch.nn.init as init

# --- 核心改动：从注册器导入分支 ---
from models.submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from models.submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY

# --- 特征融合模块 (保持不变) ---
class FusionModule(nn.Module):
    def __init__(self, config):
        super(FusionModule, self).__init__()
        self.strategy = config.get('strategy', 'concat').lower()
        dim_norm, dim_cont = config['dim_norm'], config['dim_cont']
        if self.strategy == 'add':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.output_dim = dim_norm
        elif self.strategy == 'attention':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.attention = nn.MultiheadAttention(dim_norm, config.get('attention_heads', 4), batch_first=True)
            self.output_dim = dim_norm
        else: self.output_dim = dim_norm + dim_cont

    def forward(self, features_norm, features_cont):
        features_norm = features_norm.transpose(1, 2)
        features_cont_expanded = features_cont.unsqueeze(1).expand(-1, features_norm.size(1), -1)
        if self.strategy == 'add': return features_norm + self.project_cont(features_cont_expanded)
        elif self.strategy == 'attention':
            projected_cont = self.project_cont(features_cont_expanded)
            return self.attention(features_norm, projected_cont, projected_cont, need_weights=False)[0]
        else: return torch.cat([features_norm, features_cont_expanded], dim=-1)

# --- 主模型: CustomFusionNet (使用注册器重构) ---
@register_model('CustomFusionNet')
class CustomFusionNet(nn.Module):
    def __init__(self, configs):
        super(CustomFusionNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        # --- 使用注册器动态构建分支 ---
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.normalized_branch = NormBranchClass(model_config['normalized_branch_config'])
        
        if self.task_name == 'spectral_prediction':
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
            cont_branch_config = model_config['continuum_branch_config']
            cont_branch_config['feature_size'] = configs.feature_size
            self.continuum_branch = ContBranchClass(cont_branch_config)
            
            fusion_config = model_config['fusion_config']
            fusion_config['dim_norm'] = self.normalized_branch.output_dim
            fusion_config['dim_cont'] = self.continuum_branch.output_dim
            self.fusion = FusionModule(fusion_config)
            lstm_input_dim = self.fusion.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.normalized_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

        lstm_conf = model_config['fusion_config']
        self.bilstm = nn.LSTM(lstm_input_dim, lstm_conf['lstm_hidden_dim'], lstm_conf['lstm_layers'], batch_first=True, bidirectional=True, dropout=lstm_conf.get('dropout', 0.2))

        head_input_dim = lstm_conf['lstm_hidden_dim'] * 2
        self.prediction_heads = nn.ModuleDict()
        for target in self.targets:
            layers, in_features = [], head_input_dim
            for out_features in model_config['prediction_head_config']['hidden_layers']:
                layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
                in_features = out_features
            layers.append(nn.Linear(in_features, 1))
            self.prediction_heads[target] = nn.Sequential(*layers)

    # ... forward 方法保持不变 ...
    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        lstm_out, _ = self.bilstm(fused_sequence)
        final_features = lstm_out[:, -1, :]
        return torch.cat([head(final_features) for head in self.prediction_heads.values()], dim=1)

    def forward_regression(self, x):
        features = self.normalized_branch(x)
        features = features.transpose(1, 2)
        lstm_out, _ = self.bilstm(features)
        final_features = lstm_out[:, -1, :]
        return torch.cat([head(final_features) for head in self.prediction_heads.values()], dim=1)