
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

from . import register_normalized_branch

@register_normalized_branch
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
