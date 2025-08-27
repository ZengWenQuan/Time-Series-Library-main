import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_continuum_branch
from ..normalized_branches.multiscale_pyramid_branch import MultiScalePyramidBranch

class FrequencyFeatureExtractor(nn.Module):
    """封装频域特征提取逻辑"""
    def __init__(self, fft_params):
        super().__init__()
        self.n_fft = fft_params.get('n_fft')

    def forward(self, x):
        # Expects (B, 1, L) or (B, L). Squeeze to (B, L) for FFT.
        if x.dim() == 3:
            x_squeezed = x.squeeze(1)
        else:
            x_squeezed = x
        
        x_fft = torch.fft.rfft(x_squeezed, n=self.n_fft)
        return torch.stack([x_fft.real, x_fft.imag], dim=1)

class AttentionGate(nn.Module):
    """根据频域特征生成通道注意力权重"""
    def __init__(self, config):
        super(AttentionGate, self).__init__()
        attention_channels = config['attention_channels']
        conv_config = config['gating_conv_layers']
        in_channels = 2
        conv_layers = []
        current_len = config['fft_len']

        for layer_cfg in conv_config:
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg['kernel_size']
            stride = layer_cfg.get('stride', 1)
            padding = layer_cfg.get('padding', (kernel_size - 1) // 2)
            pool_size = layer_cfg.get('pool_size', 2)
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            if pool_size > 1:
                conv_layers.append(nn.MaxPool1d(pool_size))
            in_channels = out_channels
            current_len = (current_len + 2 * padding - (kernel_size - 1) - 1) // stride + 1
            if pool_size > 1:
                current_len = (current_len - (pool_size - 1) - 1) // pool_size + 1

        self.conv_block = nn.Sequential(*conv_layers)
        fnn_input_dim = in_channels * current_len
        fnn_hidden_dims = config.get('gating_hidden_dims', [])
        fnn_layers = []
        current_dim = fnn_input_dim
        for h_dim in fnn_hidden_dims:
            fnn_layers.append(nn.Linear(current_dim, h_dim))
            fnn_layers.append(nn.ReLU(inplace=True))
            current_dim = h_dim
        
        fnn_layers.append(nn.Linear(current_dim, attention_channels))
        fnn_layers.append(nn.Sigmoid())
        self.gate_fnn = nn.Sequential(*fnn_layers)

    def forward(self, x):
        x_conv = self.conv_block(x)
        x_flat = x_conv.view(x_conv.size(0), -1)
        attention_weights = self.gate_fnn(x_flat)
        return attention_weights

@register_continuum_branch
class CustomAttentiveBranch(nn.Module):
    """
    A convolutional branch with a parallel channel attention mechanism.
    The attention weights are derived from frequency-domain features.
    """
    def __init__(self, config):
        super(CustomAttentiveBranch, self).__init__()
        
        # 1. Main convolutional path
        self.main_branch = MultiScalePyramidBranch(config['main_branch_config'])
        main_branch_out_channels = self.main_branch.output_channels

        # 2. Attention path
        self.feature_extractor = FrequencyFeatureExtractor(config['fft'])
        
        gate_config = config['attention_gate_config']
        gate_config['fft_len'] = config['fft']['n_fft'] // 2 + 1
        gate_config['attention_channels'] = main_branch_out_channels
        self.attention_gate = AttentionGate(gate_config)

        # 3. Define overall output shape
        self.output_channels = self.main_branch.output_channels
        self.output_length = self.main_branch.output_length

    def forward(self, x):
        # Main path
        main_features = self.main_branch(x) # (B, C, L)
        
        # Attention path
        fft_features = self.feature_extractor(x)
        channel_weights = self.attention_gate(fft_features) # (B, C)
        
        # Apply attention
        output = main_features * channel_weights.unsqueeze(-1)
        
        return output