import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model
import torch.nn.init as init

# --- 核心改动：从注册器导入分支 ---
from models.submodules.continuum_branches import CONTINUUM_BRANCH_REGISTRY
from models.submodules.normalized_branches import NORMALIZED_BRANCH_REGISTRY

# --- 主模型: LargeKernelConvNet (使用注册器重构) ---
@register_model('LargeKernelConvNet')
class LargeKernelConvNet(nn.Module):
    def __init__(self, configs):
        super(LargeKernelConvNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        # --- 使用注册器动态构建分支 ---
        NormBranchClass = NORMALIZED_BRANCH_REGISTRY[model_config['normalized_branch_name']]
        self.upsample_branch = NormBranchClass(model_config['upsample_branch_config'])
        
        if self.task_name == 'spectral_prediction':
            ContBranchClass = CONTINUUM_BRANCH_REGISTRY[model_config['continuum_branch_name']]
            self.large_kernel_branch = ContBranchClass(model_config['large_kernel_branch_config'])
            lstm_input_dim = self.upsample_branch.output_dim + self.large_kernel_branch.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.upsample_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

        # --- 后续逻辑保持不变 ---
        lstm_conf = model_config['bilstm_head_config']
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
        features_cont_vec = self.large_kernel_branch(x_continuum)
        features_norm_map = self.upsample_branch(x_normalized)
        features_norm_seq = features_norm_map.transpose(1, 2)
        cont_expanded = features_cont_vec.unsqueeze(1).expand(-1, features_norm_seq.size(1), -1)
        fused_sequence = torch.cat([features_norm_seq, cont_expanded], dim=-1)
        lstm_out, _ = self.bilstm(fused_sequence)
        final_features = lstm_out[:, -1, :]
        return torch.cat([head(final_features) for head in self.prediction_heads.values()], dim=1)

    def forward_regression(self, x):
        features_map = self.upsample_branch(x)
        features_seq = features_map.transpose(1, 2)
        lstm_out, _ = self.bilstm(features_seq)
        final_features = lstm_out[:, -1, :]
        return torch.cat([head(final_features) for head in self.prediction_heads.values()], dim=1)