
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

# --- 2. 归一化谱/通用特征的多尺度处理分支 ---
class InceptionBlock(nn.Module):
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
    def __init__(self, feature_size, config):
        super(NormalizedSpectrumBranch, self).__init__()

        # --- 新增：根据网络配置，检查输入尺寸是否有效 ---
        total_downsample_factor = 1
        for layer_conf in config['layers']:
            total_downsample_factor *= layer_conf.get('pool_size', 1)
        
        if feature_size < total_downsample_factor:
            raise ValueError(
                f"Input feature_size ({feature_size}) is too small for the configured pooling layers. "
                f"With the current architecture, the minimum feature_size is {total_downsample_factor}."
            )
        # --- 检查结束 ---

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
        x = x.unsqueeze(1)
        return self.network(x)

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
        features_norm = features_norm.transpose(1, 2)
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
        self.task_name = configs.task_name
        self.targets = configs.targets
        
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # --- 初始化所有可能的模块 ---
        self.normalized_branch = NormalizedSpectrumBranch(configs.feature_size, model_config['normalized_branch'])
        
        if self.task_name == 'spectral_prediction':
            self.continuum_branch = FrequencyBranch(configs.feature_size, model_config['continuum_freq_branch'])
            self.fusion = FusionModule(model_config['fusion'], self.normalized_branch.output_dim, self.continuum_branch.output_dim)
            lstm_input_dim = self.fusion.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.normalized_branch.output_dim
        else:
            raise ValueError(f"Task name '{self.task_name}' is not supported by CustomFusionNet.")

        lstm_conf = model_config['fusion']
        self.bilstm = nn.LSTM(lstm_input_dim, lstm_conf['lstm_hidden_dim'], lstm_conf['lstm_layers'], 
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

    def forward(self, x):
        # 这个 forward 方法现在作为路由器，根据任务和输入参数来决定执行路径

        # 场景一：回归任务 (总是接收一个2D张量x, x_normalized为None)
        if self.task_name == 'regression':
            return self.forward_regression(x)

        # 场景二：光谱预测任务
        elif self.task_name == 'spectral_prediction':
            # 如果 x_normalized 不是 None，说明是旧的调用方式，分别传入了两个特征
            x_continuum = x[:,:, 0]
            x_normalized = x[:,:,1]
            return self.forward_spectral_prediction(x_continuum, x_normalized)
        else:
            raise ValueError(f"Task name '{self.task_name}' is not supported by CustomFusionNet.")

    def forward_spectral_prediction(self, x_continuum, x_normalized):
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1: x_continuum = x_continuum.squeeze(-1)
        if x_normalized.ndim == 3 and x_normalized.shape[2] == 1: x_normalized = x_normalized.squeeze(-1)

        features_norm = self.normalized_branch(x_normalized)
        features_cont = self.continuum_branch(x_continuum)
        fused_sequence = self.fusion(features_norm, features_cont)
        lstm_out, _ = self.bilstm(fused_sequence)
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        return torch.cat(predictions, dim=1)

    def forward_regression(self, x):
        if x.ndim == 3 and x.shape[2] == 1: x = x.squeeze(-1)
        
        features = self.normalized_branch(x) # [B, D_norm, L_down]
        features = features.transpose(1, 2) # [B, L_down, D_norm]
        lstm_out, _ = self.bilstm(features)
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        return torch.cat(predictions, dim=1)
