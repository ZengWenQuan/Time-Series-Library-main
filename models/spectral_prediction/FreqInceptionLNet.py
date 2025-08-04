
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 辅助模块：用于卷积层的LayerNorm ---
class LayerNormForConv(nn.Module):
    """对Conv1d的输出 (N, C, L) 进行LayerNorm的标准实现"""
    def __init__(self, num_channels, eps, elementwise_affine):
        super(LayerNormForConv, self).__init__()
        # LayerNorm作用于最后一个维度，所以我们需要将channel维度换到最后
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x shape: (N, C, L)
        # transpose to (N, L, C) for LayerNorm
        x = x.transpose(1, 2)
        # apply LayerNorm
        x = self.ln(x)
        # transpose back to (N, C, L)
        x = x.transpose(1, 2)
        return x

# --- 1. 通道注意力机制 (SE Block) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- 2. 频率分支 ---
class FrequencyBranch(nn.Module):
    def __init__(self, config):
        super(FrequencyBranch, self).__init__()
        branch_config = config['frequency_branch']
        norm_settings = config['normalization_settings']
        dropout_rate = branch_config['dropout']
        in_channels = 2
        
        self.cnn = nn.ModuleList()
        for layer_conf in branch_config['conv_layers']:
            self.cnn.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_conf['out_channels'],
                kernel_size=layer_conf['kernel_size'],
                stride=layer_conf['stride'],
                padding=layer_conf['padding']
            ))
            if norm_settings['norm_type'] == 'bn':
                self.cnn.append(nn.BatchNorm1d(layer_conf['out_channels']))
            elif norm_settings['norm_type'] == 'ln':
                self.cnn.append(LayerNormForConv(layer_conf['out_channels'], norm_settings['layer_norm_eps'], norm_settings['layer_norm_elementwise_affine']))
            self.cnn.append(nn.ReLU(inplace=True))
            in_channels = layer_conf['out_channels']
        
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, branch_config['mlp_output_dim']),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        fft_result = torch.fft.rfft(x, norm='ortho')
        fft_features = torch.stack([fft_result.real, fft_result.imag], dim=1)
        
        for layer in self.cnn:
            fft_features = layer(fft_features)

        pooled_output = self.final_pool(fft_features).squeeze(-1)
        output = self.mlp(pooled_output)
        return output

# --- 3. Inception分支 ---
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_map, reduction_ratio, norm_settings):
        super(InceptionBlock, self).__init__()
        self.paths = nn.ModuleList()
        total_out_channels = 0

        for kernel_size, out_channels in out_channels_map.items():
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)]
            if norm_settings['norm_type'] == 'bn':
                layers.append(nn.BatchNorm1d(out_channels))
            elif norm_settings['norm_type'] == 'ln':
                layers.append(LayerNormForConv(out_channels, norm_settings['layer_norm_eps'], norm_settings['layer_norm_elementwise_affine']))
            layers.append(nn.ReLU(inplace=True))
            self.paths.append(nn.Sequential(*layers))
            total_out_channels += out_channels

        self.attention = ChannelAttention(total_out_channels, reduction_ratio)

    def forward(self, x):
        path_outputs = [path(x) for path in self.paths]
        concatenated = torch.cat(path_outputs, dim=1)
        attended = self.attention(concatenated)
        return attended

class InceptionBranch(nn.Module):
    def __init__(self, config):
        super(InceptionBranch, self).__init__()
        branch_config = config['inception_branch']
        norm_settings = config['normalization_settings']
        self.blocks = nn.ModuleList()
        in_channels = 1

        for block_config in branch_config['blocks']:
            self.blocks.append(InceptionBlock(
                in_channels=in_channels,
                out_channels_map=block_config['out_channels_per_path'],
                reduction_ratio=block_config['channel_attention_reduction'],
                norm_settings=norm_settings
            ))
            in_channels = sum(block_config['out_channels_per_path'].values())

            if block_config['use_pooling']:
                self.blocks.append(nn.MaxPool1d(kernel_size=block_config['pool_size']))
        
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = in_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        output = self.final_pool(x).squeeze(-1)
        return output

# --- 4. 主模型 FreqInceptionLNet ---
@register_model('FreqInceptionLNet')
class FreqInceptionLNet(nn.Module):
    def __init__(self, configs):
        super(FreqInceptionLNet, self).__init__()
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size

        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)
        
        model_config['feature_size'] = self.feature_size
        self.norm_type = model_config['normalization_settings']['norm_type']

        self.frequency_branch = FrequencyBranch(model_config)
        self.inception_branch = InceptionBranch(model_config)

        fusion_input_dim = model_config['frequency_branch']['mlp_output_dim'] + self.inception_branch.output_dim
        self.fusion_dropout = nn.Dropout(model_config['fusion_module']['dropout'])

        lstm_config = model_config['sequence_module']
        self.bilstm = nn.LSTM(
            input_size=fusion_input_dim,
            hidden_size=lstm_config['lstm_hidden_dim'],
            num_layers=lstm_config['lstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=lstm_config['lstm_dropout'] if lstm_config['lstm_layers'] > 1 else 0
        )

        self.prediction_head = self._build_prediction_head(model_config)
        self._print_model_info(model_config)

    def _build_prediction_head(self, config):
        head_config = config['prediction_head']
        lstm_config = config['sequence_module']
        norm_settings = config['normalization_settings']
        ffn_layers = []
        in_dim = lstm_config['lstm_hidden_dim'] * 2
        for hidden_dim in head_config['hidden_dims']:
            ffn_layers.append(nn.Linear(in_dim, hidden_dim))
            if norm_settings['norm_type'] == 'bn':
                ffn_layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_settings['norm_type'] == 'ln':
                # 在FFN中，可以直接使用标准的LayerNorm
                ffn_layers.append(nn.LayerNorm(hidden_dim, eps=norm_settings['layer_norm_eps'], elementwise_affine=norm_settings['layer_norm_elementwise_affine']))
            ffn_layers.append(nn.ReLU(inplace=True))
            ffn_layers.append(nn.Dropout(head_config['dropout']))
            in_dim = hidden_dim
        ffn_layers.append(nn.Linear(in_dim, self.label_size))
        return nn.Sequential(*ffn_layers)

    def _print_model_info(self, config):
        total_params = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print(f"FreqInceptionLNet Model Initialized")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Normalization Type: {self.norm_type.upper()}")
        print("=" * 60)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'spectral_prediction':
            return self.regression(x_enc)
        raise ValueError(f"Task name '{self.task_name}' not supported.")

    def regression(self, x_enc):
        continuum_spec = x_enc[:, :self.feature_size]
        absorption_spec = x_enc[:, self.feature_size:]

        freq_features = self.frequency_branch(continuum_spec)
        inception_features = self.inception_branch(absorption_spec)

        fused_features = torch.cat((freq_features, inception_features), dim=1)
        fused_features = self.fusion_dropout(fused_features)

        lstm_input = fused_features.unsqueeze(1)
        lstm_output, _ = self.bilstm(lstm_input)
        sequence_features = lstm_output.squeeze(1)

        predictions = self.prediction_head(sequence_features)
        return predictions
