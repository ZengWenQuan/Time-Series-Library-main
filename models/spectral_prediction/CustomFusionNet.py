import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 1. 连续谱的频域处理分支 ---
class FrequencyBranch(nn.Module):
    def __init__(self, feature_size, config):
        super(FrequencyBranch, self).__init__()
        fft_len = feature_size // 2 + 1
        layers = []
        in_features = fft_len
        for out_features in config['mlp_layers']:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        self.mlp = nn.Sequential(*layers)
        self.output_dim = in_features

    def forward(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho')
        x_fft_mag = x_fft.abs()
        return self.mlp(x_fft_mag)

# --- 2. 归一化谱的多尺度处理分支 (全新实现) ---
class InceptionBlock(nn.Module):
    """一个并行的多尺度卷积块"""
    def __init__(self, in_channels, out_channels_per_path):
        super(InceptionBlock, self).__init__()
        self.path1 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=1, padding='same')
        self.path2 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=3, padding='same')
        self.path3 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=5, padding='same')
        self.output_channels = out_channels_per_path * 3

    def forward(self, x):
        x1 = F.relu(self.path1(x))
        x2 = F.relu(self.path2(x))
        x3 = F.relu(self.path3(x))
        return torch.cat([x1, x2, x3], dim=1)

class NormalizedSpectrumBranch(nn.Module):
    """受 mspdownsample.py 思想启发，全新实现的多尺度处理分支"""
    def __init__(self, config):
        super(NormalizedSpectrumBranch, self).__init__()
        layers = []
        in_channels = 1
        for layer_conf in config['layers']:
            block = InceptionBlock(in_channels, layer_conf['out_channels_per_path'])
            layers.append(block)
            layers.append(nn.BatchNorm1d(block.output_channels))
            layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = block.output_channels
        self.network = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        x = x.unsqueeze(1) # [B, L] -> [B, 1, L]
        return self.network(x) # [B, D_norm, L_down]

# --- 3. 特征融合模块 ---
class FusionModule(nn.Module):
    def __init__(self, config, dim_norm, dim_cont):
        super(FusionModule, self).__init__()
        self.strategy = config.get('strategy', 'concat').lower()

        if self.strategy == 'add':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.output_dim = dim_norm
        elif self.strategy == 'attention':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.attention = nn.MultiheadAttention(embed_dim=dim_norm, num_heads=config.get('attention_heads', 4), batch_first=True)
            self.output_dim = dim_norm
        else: # concat
            self.output_dim = dim_norm + dim_cont

    def forward(self, features_norm, features_cont):
        # features_norm: [B, D_norm, L_down], features_cont: [B, D_cont]
        features_norm = features_norm.transpose(1, 2) # -> [B, L_down, D_norm]
        features_cont_expanded = features_cont.unsqueeze(1).expand(-1, features_norm.size(1), -1)

        if self.strategy == 'add':
            return features_norm + self.project_cont(features_cont_expanded)
        elif self.strategy == 'attention':
            projected_cont = self.project_cont(features_cont_expanded)
            attn_output, _ = self.attention(query=features_norm, key=projected_cont, value=projected_cont)
            return attn_output
        else: # concat
            return torch.cat([features_norm, features_cont_expanded], dim=-1)

# --- 4. 主模型: CustomFusionNet ---
@register_model('CustomFusionNet')
class CustomFusionNet(nn.Module):
    def __init__(self, configs):
        super(CustomFusionNet, self).__init__()
        self.targets = configs.targets
        
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        self.normalized_branch = NormalizedSpectrumBranch(model_config['normalized_branch'])
        self.continuum_branch = FrequencyBranch(configs.feature_size, model_config['continuum_freq_branch'])
        self.fusion = FusionModule(model_config['fusion'], self.normalized_branch.output_dim, self.continuum_branch.output_dim)

        lstm_conf = model_config['fusion']
        self.bilstm = nn.LSTM(self.fusion.output_dim, lstm_conf['lstm_hidden_dim'], lstm_conf['lstm_layers'], 
                              batch_first=True, bidirectional=True, dropout=lstm_conf.get('dropout', 0.2))

        head_input_dim = lstm_conf['lstm_hidden_dim'] * 2
        self.prediction_heads = nn.ModuleDict()
        for target in self.targets:
            head_layers = []
            in_features = head_input_dim
            for out_features in model_config['prediction_head']['hidden_layers']:
                head_layers.append(nn.Linear(in_features, out_features))
                head_layers.append(nn.ReLU())
                in_features = out_features
            head_layers.append(nn.Linear(in_features, 1))
            self.prediction_heads[target] = nn.Sequential(*head_layers)

    def forward(self, x_continuum, x_normalized):
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1:
            x_continuum = x_continuum.squeeze(-1)
        if x_normalized.ndim == 3 and x_normalized.shape[2] == 1:
            x_normalized = x_normalized.squeeze(-1)

        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)

        fused_sequence = self.fusion(features_norm, features_cont)
        lstm_out, _ = self.bilstm(fused_sequence)
        
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        
        return torch.cat(predictions, dim=1)