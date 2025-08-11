import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model
import torch.nn.init as init

# --- 1. 连续谱的频域处理分支 (保持不变) ---
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

# --- 2. 归一化谱处理分支 (按 msp.txt 重写) ---
class PyramidBlock(nn.Module):
    """金字塔块：三分支并行处理，模仿 msp.txt 设计"""
    def __init__(self, input_channel, output_channel, kernel_sizes, use_batch_norm, use_attention, attention_reduction):
        super(PyramidBlock, self).__init__()
        self.use_attention = use_attention
        self.fine_branch = self._make_branch(input_channel, output_channel, kernel_sizes[0], use_batch_norm)
        self.medium_branch = self._make_branch(input_channel, output_channel, kernel_sizes[1], use_batch_norm)
        self.coarse_branch = self._make_branch(input_channel, output_channel, kernel_sizes[2], use_batch_norm)
        
        self.output_channels = output_channel * 3
        self.residual = nn.Sequential()
        if input_channel != self.output_channels:
            layers = [nn.Conv1d(input_channel, self.output_channels, kernel_size=1, bias=not use_batch_norm)]
            if use_batch_norm: layers.append(nn.BatchNorm1d(self.output_channels))
            self.residual = nn.Sequential(*layers)
            
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.output_channels, max(1, self.output_channels // attention_reduction), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, self.output_channels // attention_reduction), self.output_channels, kernel_size=1),
                nn.Sigmoid()
            )
        self._initialize_weights()
    
    def _make_branch(self, in_ch, out_ch, ks, use_bn):
        padding = ks // 2
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=ks, padding=padding, bias=not use_bn),
            nn.BatchNorm1d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        fine_out = self.fine_branch(x)
        medium_out = self.medium_branch(x)
        coarse_out = self.coarse_branch(x)
        pyramid_out = torch.cat([fine_out, medium_out, coarse_out], dim=1)
        if self.use_attention:
            pyramid_out = pyramid_out * self.attention(pyramid_out)
        return pyramid_out + self.residual(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d): init.constant_(m.weight, 1); init.constant_(m.bias, 0)

class NormalizedSpectrumBranch(nn.Module):
    """模仿 MSPNet 结构，用于提取归一化谱特征"""
    def __init__(self, config):
        super(NormalizedSpectrumBranch, self).__init__()
        use_bn = config['batch_norm']
        kernel_sizes = config['kernel_sizes']
        pyramid_channels = config['pyramid_channels']
        pool_size = config['pool_size']
        use_attention = config['use_attention']
        attention_reduction = config['attention_reduction']

        self.input_proj = nn.Sequential(
            nn.Conv1d(1, pyramid_channels[0], kernel_size=7, padding=3, bias=not use_bn),
            nn.BatchNorm1d(pyramid_channels[0]) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.pyramid_blocks = nn.ModuleList()
        in_ch = pyramid_channels[0]
        for i, out_ch in enumerate(pyramid_channels):
            self.pyramid_blocks.append(PyramidBlock(in_ch, out_ch, kernel_sizes, use_bn, use_attention, attention_reduction))
            self.pyramid_blocks.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            in_ch = out_ch * 3
        
        self.output_dim = in_ch

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        for block in self.pyramid_blocks:
            x = block(x)
        return x

# --- 3. 特征融合模块 (保持不变) ---
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
        else: self.output_dim = dim_norm + dim_cont

    def forward(self, features_norm, features_cont):
        features_norm = features_norm.transpose(1, 2)
        features_cont_expanded = features_cont.unsqueeze(1).expand(-1, features_norm.size(1), -1)
        if self.strategy == 'add': return features_norm + self.project_cont(features_cont_expanded)
        elif self.strategy == 'attention':
            projected_cont = self.project_cont(features_cont_expanded)
            attn_output, _ = self.attention(query=features_norm, key=projected_cont, value=projected_cont)
            return attn_output
        else: return torch.cat([features_norm, features_cont_expanded], dim=-1)

# --- 4. 主模型: CustomFusionNet (保持不变) ---
@register_model('CustomFusionNet')
class CustomFusionNet(nn.Module):
    def __init__(self, configs):
        super(CustomFusionNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        self.normalized_branch = NormalizedSpectrumBranch(model_config['normalized_branch'])
        if self.task_name == 'spectral_prediction':
            self.continuum_branch = FrequencyBranch(configs.feature_size, model_config['continuum_freq_branch'])
            self.fusion = FusionModule(model_config['fusion'], self.normalized_branch.output_dim, self.continuum_branch.output_dim)
            lstm_input_dim = self.fusion.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.normalized_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported by CustomFusionNet.")

        lstm_conf = model_config['fusion']
        self.bilstm = nn.LSTM(lstm_input_dim, lstm_conf['lstm_hidden_dim'], lstm_conf['lstm_layers'], batch_first=True, bidirectional=True, dropout=lstm_conf.get('dropout', 0.2))

        head_input_dim = lstm_conf['lstm_hidden_dim'] * 2
        self.prediction_heads = nn.ModuleDict()
        for target in self.targets:
            layers, in_features = [], head_input_dim
            for out_features in model_config['prediction_head']['hidden_layers']:
                layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
                in_features = out_features
            layers.append(nn.Linear(in_features, 1))
            self.prediction_heads[target] = nn.Sequential(*layers)

    def forward(self, x, x_normalized=None):
        if self.task_name == 'spectral_prediction':
            if x_normalized is None: x_continuum, x_normalized = x[:, :, 0], x[:, :, 1]
            else: x_continuum = x
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        elif self.task_name == 'regression': return self.forward_regression(x)
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        lstm_out, _ = self.bilstm(fused_sequence)
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        return torch.cat(predictions, dim=1)

    def forward_regression(self, x):
        features = self.normalized_branch(x)
        features = features.transpose(1, 2)
        lstm_out, _ = self.bilstm(features)
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        return torch.cat(predictions, dim=1)