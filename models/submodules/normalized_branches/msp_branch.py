
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class PyramidBlock(nn.Module):
    def __init__(self, config):
        super(PyramidBlock, self).__init__()
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

from ...registries import register_normalized_branch

@register_normalized_branch
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
