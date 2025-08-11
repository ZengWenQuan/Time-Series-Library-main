import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model
import torch.nn.init as init

# --- 辅助模块：从 msp.txt 参考来的金字塔块 ---
class PyramidBlock(nn.Module):
    def __init__(self, config):
        super(PyramidBlock, self).__init__()
        use_bn = config['batch_norm']
        self.use_attention = config['use_attention']
        self.fine_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][0], use_bn)
        self.medium_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][1], use_bn)
        self.coarse_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][2], use_bn)
        
        self.output_channels = config['output_channel'] * 3
        self.residual = nn.Sequential()
        if config['input_channel'] != self.output_channels:
            layers = [nn.Conv1d(config['input_channel'], self.output_channels, 1, bias=not use_bn)]
            if use_bn: layers.append(nn.BatchNorm1d(self.output_channels))
            self.residual = nn.Sequential(*layers)
            
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(self.output_channels, max(1, self.output_channels // config['attention_reduction']), 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, self.output_channels // config['attention_reduction']), self.output_channels, 1),
                nn.Sigmoid())
        self._initialize_weights()
    
    def _make_branch(self, in_ch, out_ch, ks, use_bn):
        layers = [nn.Conv1d(in_ch, out_ch, ks, padding=ks//2, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        pyramid_out = torch.cat([self.fine_branch(x), self.medium_branch(x), self.coarse_branch(x)], dim=1)
        if self.use_attention: pyramid_out = pyramid_out * self.attention(pyramid_out)
        return pyramid_out + self.residual(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d): init.constant_(m.weight, 1); init.constant_(m.bias, 0)

# --- 1. 超大核卷积分支 (用于连续谱) ---
class LargeKernelBranch(nn.Module):
    def __init__(self, config):
        super(LargeKernelBranch, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=(config['kernel_size'] - 1) // 2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config['out_channels'], config['fc_dim'])
        self.output_dim = config['fc_dim']

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- 2. 上采样+多尺度分支 (用于归一化谱或回归任务) ---
class UpsampleMultiScaleBranch(nn.Module):
    def __init__(self, config):
        super(UpsampleMultiScaleBranch, self).__init__()
        self.upsample_conv = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=config['pyramid_channels'][0],
            kernel_size=config['upsample_kernel'],
            stride=2,
            padding=(config['upsample_kernel'] - 2) // 2
        )
        self.pyramid_blocks = nn.ModuleList()
        in_ch = config['pyramid_channels'][0]
        for out_ch in config['pyramid_channels']:
            block_config = config.copy()
            block_config.update({'input_channel': in_ch, 'output_channel': out_ch})
            self.pyramid_blocks.append(PyramidBlock(block_config))
            self.pyramid_blocks.append(nn.AvgPool1d(kernel_size=config['pool_size']))
            in_ch = out_ch * 3
        self.output_dim = in_ch

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.upsample_conv(x))
        for block in self.pyramid_blocks:
            x = block(x)
        return x

# --- 3. 主模型: LargeKernelConvNet ---
@register_model('LargeKernelConvNet')
class LargeKernelConvNet(nn.Module):
    def __init__(self, configs):
        super(LargeKernelConvNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        self.upsample_branch = UpsampleMultiScaleBranch(model_config['upsample_branch'])
        if self.task_name == 'spectral_prediction':
            self.large_kernel_branch = LargeKernelBranch(model_config['large_kernel_branch'])
            lstm_input_dim = self.upsample_branch.output_dim + self.large_kernel_branch.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.upsample_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

        lstm_conf = model_config['bilstm_head']
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