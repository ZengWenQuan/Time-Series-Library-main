
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_normalized_branch

class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, use_attention, use_batch_norm, dropout_rate):
        super(MultiScaleBlock, self).__init__()
        self.use_attention = use_attention
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)]
            if use_batch_norm: layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.branches.append(nn.Sequential(*layers))
        
        total_channels = out_channels * len(kernel_sizes)
        fusion_layers = [nn.Conv1d(total_channels, out_channels, kernel_size=1, bias=True)]
        if use_batch_norm: fusion_layers.append(nn.BatchNorm1d(out_channels))
        fusion_layers.extend([nn.ReLU(inplace=True), nn.Dropout(dropout_rate)])
        self.fusion = nn.Sequential(*fusion_layers)
        
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            residual_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)]
            if use_batch_norm: residual_layers.append(nn.BatchNorm1d(out_channels))
            self.residual = nn.Sequential(*residual_layers)
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(out_channels, max(1, out_channels // 8), 1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, out_channels // 8), out_channels, 1),
                nn.Sigmoid())
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        fused_features = self.fusion(torch.cat(branch_outputs, dim=1))
        if self.use_attention: fused_features = fused_features * self.attention(fused_features)
        return F.relu(fused_features + self.residual(x))

@register_normalized_branch
class AbsorptionBranch(nn.Module):
    def __init__(self, config):
        super(AbsorptionBranch, self).__init__()
        absorption_config = config['blocks']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.1)
        
        self.multiscale_blocks = nn.ModuleList()
        in_channels = 1
        for block_config in absorption_config:
            block = MultiScaleBlock(in_channels, block_config['out_channels'], block_config['kernel_sizes'], 
                                  block_config.get('use_attention', False), use_batch_norm, dropout_rate)
            self.multiscale_blocks.append(block)
            if block_config.get('downsample', False):
                self.multiscale_blocks.append(nn.AvgPool1d(block_config.get('pool_size', 2), stride=block_config.get('pool_stride', 2)))
            in_channels = block_config['out_channels']
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = in_channels
        
    def forward(self, x):
        x = x.unsqueeze(1)
        for block in self.multiscale_blocks: x = block(x)
        x = self.global_pool(x).squeeze(-1)
        return x
