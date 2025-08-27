
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from ...registries import register_continuum_branch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe * 0.1
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return torch.clamp(x + self.pe[:x.size(0), :], min=-10.0, max=10.0)

@register_continuum_branch
class ContinuumBranch(nn.Module):
    def __init__(self, config):
        super(ContinuumBranch, self).__init__()
        cnn_config = config['cnn']
        trans_config = config['transformer']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.1)
        
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for layer_config in cnn_config['layers']:
            layers = [nn.Conv1d(in_channels, layer_config['out_channels'], layer_config['kernel_size'], layer_config['stride'], layer_config['padding'], bias=True)]
            if use_batch_norm: layers.append(nn.BatchNorm1d(layer_config['out_channels']))
            layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])
            self.cnn_layers.append(nn.Sequential(*layers))
            in_channels = layer_config['out_channels']
        
        self.feature_projection = nn.Linear(in_channels, trans_config['d_model'])
        self.pos_encoder = PositionalEncoding(trans_config['d_model'])
        self.norm = nn.LayerNorm(trans_config['d_model'], eps=1e-6)
        encoder_layer = nn.TransformerEncoderLayer(trans_config['d_model'], trans_config['n_heads'], trans_config['ffn_dim'], max(0.05, trans_config['dropout']), batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_config['num_layers'])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = trans_config['d_model']
        
    def forward(self, x):
        for cnn_layer in self.cnn_layers: x = cnn_layer(x)
        x = x.permute(0, 2, 1)
        x = self.feature_projection(x)
        x_pos = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.norm(x_pos)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        return x
