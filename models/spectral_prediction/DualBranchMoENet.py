import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 核心改动：从注册器导入分支 ---
from models.submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from models.submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY

# --- 主模型 ---
@register_model('DualBranchMoENet')
class DualBranchMoENet(nn.Module):
    def __init__(self, configs):
        super(DualBranchMoENet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets

        with open(configs.model_conf, 'r') as f:
            config = yaml.safe_load(f)
        
        norm_type = config['normalization_type']

        # --- 使用注册器动态构建分支 ---
        ContBranchClass = CONTINUUM_BRANCH_REGISTRY[config['continuum_branch_name']]
        self.freq_branch = ContBranchClass(config['continuum_branch_config'], norm_type)

        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[config['normalized_branch_name']]
        self.line_branch = NormBranchClass(config['normalized_branch_config']['pyramid_with_attention'], norm_type)

        # --- 后续逻辑保持不变 ---
        fusion_conf = config['fusion_module_config']
        
        if self.task_name == 'spectral_prediction':
            lstm_input_dim = self.freq_branch.output_dim + self.line_branch.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.line_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

        self.fusion_lstm = nn.LSTM(lstm_input_dim, fusion_conf['lstm_hidden_dim'], fusion_conf['lstm_layers'], 
                                 batch_first=True, bidirectional=True, 
                                 dropout=fusion_conf.get('dropout_rate', 0.2) if fusion_conf['lstm_layers'] > 1 else 0)
        
        ffn_layers = []
        in_dim = fusion_conf['lstm_hidden_dim'] * 2
        for layer_conf in fusion_conf['ffn']:
            ffn_layers.append(nn.Linear(in_dim, layer_conf['out_features']))
            if not layer_conf.get('is_output_layer', False):
                ffn_layers.append(nn.ReLU(inplace=True))
                ffn_layers.append(nn.Dropout(fusion_conf.get('dropout_rate', 0.2)))
            in_dim = layer_conf['out_features']
        self.prediction_head = nn.Sequential(*ffn_layers)

    # ... forward 方法保持不变 ...
    def forward(self, x, x_normalized=None):
        if self.task_name == 'regression': return self.forward_regression(x)
        elif self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1: x_continuum = x_continuum.squeeze(-1)
        freq_features = self.freq_branch(x_continuum)
        line_features = self.line_branch(x_normalized)
        len_freq, len_line = freq_features.size(2), line_features.size(2)
        if len_freq != len_line:
            target_len = min(len_freq, len_line)
            freq_features = F.interpolate(freq_features, size=target_len, mode='linear', align_corners=False)
            line_features = F.interpolate(line_features, size=target_len, mode='linear', align_corners=False)
        combined_features = torch.cat([freq_features, line_features], dim=1).permute(0, 2, 1)
        lstm_out, _ = self.fusion_lstm(combined_features)
        return self.prediction_head(lstm_out[:, -1, :])

    def forward_regression(self, x):
        features = self.line_branch(x).permute(0, 2, 1)
        lstm_out, _ = self.fusion_lstm(features)
        return self.prediction_head(lstm_out[:, -1, :])