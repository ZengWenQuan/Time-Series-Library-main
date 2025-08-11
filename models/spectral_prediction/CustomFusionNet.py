import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model
import torch.nn.init as init

try:
    from pytorch_wavelets import DWT1DForward as DWT1D
except ImportError:
    DWT1D = None

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

class ContinuumWaveletBranch(nn.Module):
    def __init__(self, config):
        super(ContinuumWaveletBranch, self).__init__()
        if DWT1D is None: raise ImportError("ContinuumWaveletBranch requires torchwavelets. Run: pip install torchwavelets")
        self.dwt = DWT1D(wave=config['wavelet_name'], J=config['wavelet_levels'], mode='symmetric')
        
        cnn_layers = []
        in_channels = 1
        for layer_conf in config['cnn']['layers']:
            block = InceptionBlock(in_channels, layer_conf['out_channels_per_path'])
            cnn_layers.append(block)
            if config.get('batch_norm', False): cnn_layers.append(nn.BatchNorm1d(block.output_channels))
            cnn_layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = block.output_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = in_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        coeffs_low, _ = self.dwt(x)
        features = self.cnn(coeffs_low)
        output = self.pool(features)
        return output.squeeze(-1)

class PyramidBlock(nn.Module):
    def __init__(self, config):
        super(PyramidBlock, self).__init__()
        # --- 修正：使用正确的键名 'batch_norm' ---
        use_batch_norm = config['batch_norm']
        self.use_attention = config['use_attention']
        self.fine_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][0], use_batch_norm)
        self.medium_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][1], use_batch_norm)
        self.coarse_branch = self._make_branch(config['input_channel'], config['output_channel'], config['kernel_sizes'][2], use_batch_norm)
        
        self.output_channels = config['output_channel'] * 3
        self.residual = nn.Sequential()
        if config['input_channel'] != self.output_channels:
            layers = [nn.Conv1d(config['input_channel'], self.output_channels, 1, bias=not use_batch_norm)]
            if use_batch_norm: layers.append(nn.BatchNorm1d(self.output_channels))
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

class NormalizedSpectrumBranch(nn.Module):
    def __init__(self, config):
        super(NormalizedSpectrumBranch, self).__init__()
        pyramid_channels = config['pyramid_channels']
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, pyramid_channels[0], 7, padding=3, bias=not config['batch_norm']),
            nn.BatchNorm1d(pyramid_channels[0]) if config['batch_norm'] else nn.Identity(),
            nn.ReLU(inplace=True))
        
        self.pyramid_blocks = nn.ModuleList()
        in_ch = pyramid_channels[0]
        for out_ch in pyramid_channels:
            block_config = config.copy()
            block_config.update({'input_channel': in_ch, 'output_channel': out_ch})
            self.pyramid_blocks.append(PyramidBlock(block_config))
            self.pyramid_blocks.append(nn.MaxPool1d(config['pool_size']))
            in_ch = out_ch * 3
        self.output_dim = in_ch

    def forward(self, x):
        x = self.input_proj(x.unsqueeze(1))
        for block in self.pyramid_blocks: x = block(x)
        return x

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

@register_model('CustomFusionNet')
class CustomFusionNet(nn.Module):
    def __init__(self, configs):
        super(CustomFusionNet, self).__init__()
        self.task_name = configs.task_name
        self.targets = configs.targets
        with open(configs.model_conf, 'r') as f: model_config = yaml.safe_load(f)

        self.normalized_branch = NormalizedSpectrumBranch(model_config['normalized_branch'])
        if self.task_name == 'spectral_prediction':
            cont_branch_config = model_config['continuum_wavelet_branch']
            cont_branch_config['feature_size'] = configs.feature_size
            self.continuum_branch = ContinuumWaveletBranch(cont_branch_config)
            
            fusion_config = model_config['fusion']
            fusion_config['dim_norm'] = self.normalized_branch.output_dim
            fusion_config['dim_cont'] = self.continuum_branch.output_dim
            self.fusion = FusionModule(fusion_config)
            lstm_input_dim = self.fusion.output_dim
        elif self.task_name == 'regression':
            lstm_input_dim = self.normalized_branch.output_dim
        else: raise ValueError(f"Task name '{self.task_name}' is not supported.")

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