
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_norm_layer(norm_type, num_features):
    if norm_type == 'batchnorm': return nn.BatchNorm1d(num_features)
    elif norm_type == 'layernorm': return nn.LayerNorm(num_features)
    else: return nn.Identity()

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class LineAttentionBranch(nn.Module):
    def __init__(self, config, norm_type):
        super(LineAttentionBranch, self).__init__()
        layers = []
        in_channels = 1
        for layer_conf in config:
            layers.append(nn.Conv1d(in_channels, layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], padding=(layer_conf['kernel_size'] - 1) // 2))
            layers.append(_get_norm_layer(norm_type, layer_conf['out_channels']))
            layers.append(nn.ReLU(inplace=True))
            if layer_conf.get('se_reduction', 0) > 0:
                layers.append(SEBlock(layer_conf['out_channels'], reduction=layer_conf['se_reduction']))
            if layer_conf.get('pool_size', 1) > 1:
                layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = layer_conf['out_channels']
        self.pyramid = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        # 修正输入维度: [B, L, 1] -> [B, L] -> [B, 1, L]
        if x.ndim == 3 and x.shape[2] == 1:
            x = x.squeeze(-1)
        return self.pyramid(x.unsqueeze(1))
