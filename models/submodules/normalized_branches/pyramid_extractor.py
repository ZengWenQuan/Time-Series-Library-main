
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from . import register_normalized_branch

class GatedActivation(nn.Module):
    def __init__(self, channels):
        super(GatedActivation, self).__init__()
        self.gate_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        return x * torch.sigmoid(self.gate_conv(x))

    def _initialize_weights(self):
        init.kaiming_normal_(self.gate_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.gate_conv.bias is not None: init.constant_(self.gate_conv.bias, 0)

class PyramidBlock(nn.Module):
    def __init__(self, config):
        super(PyramidBlock, self).__init__()
        use_bn = config['use_batch_norm']
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
            self.gate = GatedActivation(self.output_channels)
        self._initialize_weights()
    
    def _make_branch(self, in_ch, out_ch, ks, use_bn):
        layers = [nn.Conv1d(in_ch, out_ch, ks, padding=ks//2, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm1d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        pyramid_out = torch.cat([self.fine_branch(x), self.medium_branch(x), self.coarse_branch(x)], dim=1)
        if self.use_attention: pyramid_out = self.gate(pyramid_out)
        return F.relu(pyramid_out + self.residual(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d): init.constant_(m.weight, 1); init.constant_(m.bias, 0)

@register_normalized_branch
class PyramidFeatureExtractor(nn.Module):
    def __init__(self, config):
        super(PyramidFeatureExtractor, self).__init__()
        self.pyramid_channels = config['pyramid_channels']
        self.use_batch_norm = config['use_batch_norm']
        
        layers = [nn.Conv1d(1, self.pyramid_channels[0], 7, padding=3, bias=not self.use_batch_norm)]
        if self.use_batch_norm: layers.append(nn.BatchNorm1d(self.pyramid_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*layers)
        
        self.pyramid_blocks = nn.ModuleList()
        input_ch = self.pyramid_channels[0]
        for i in range(len(self.pyramid_channels)):
            block_config = config.copy()
            block_config.update({'input_channel': input_ch, 'output_channel': self.pyramid_channels[i]})
            self.pyramid_blocks.append(PyramidBlock(block_config))
            input_ch = self.pyramid_channels[i] * 3

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = input_ch

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        for block in self.pyramid_blocks:
            x = block(x)
        x = self.global_pool(x)
        return x.squeeze(-1)
