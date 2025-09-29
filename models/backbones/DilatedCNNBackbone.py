
import torch
import torch.nn as nn
from models.registries import register_backbone

class _DilatedConvBlock(nn.Module):
    """包含门控激活的空洞卷积残差块"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_batch_norm):
        super().__init__()
        
        self.conv_dilated = nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(out_channels)

        # 如果输入输出通道不匹配，需要一个投影层用于残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        
        x = self.conv_dilated(x)
        
        # Gated Activation Unit
        # 将输出通道一分为二，一个过tanh，一个过sigmoid
        out_tanh, out_sigmoid = x.chunk(2, dim=1)
        x = torch.tanh(out_tanh) * torch.sigmoid(out_sigmoid)
        
        x = self.conv_1x1(x)
        
        if self.use_batch_norm:
            x = self.norm(x)
            
        return x + shortcut

@register_backbone
class DilatedCNNBackbone(nn.Module):
    """基于空洞卷积的主干网络 (WaveNet-style)"""
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        initial_channels = cfg['initial_channels']
        block_channels = cfg['block_channels']
        num_blocks = cfg['num_blocks']
        kernel_size = cfg['kernel_size']
        use_batch_norm = cfg.get('use_batch_norm', True)

        # 初始卷积，将输入通道映射到网络内部通道
        self.initial_conv = nn.Conv1d(in_channels, initial_channels, 1)

        self.blocks = nn.ModuleList()
        current_channels = initial_channels
        for i in range(num_blocks):
            dilation = 2 ** i # 空洞率指数增长
            self.blocks.append(_DilatedConvBlock(
                current_channels, 
                block_channels, 
                kernel_size=kernel_size, 
                dilation=dilation,
                use_batch_norm=use_batch_norm
            ))
            current_channels = block_channels

        # 这个架构不改变序列长度
        self.output_length = input_length
        self.output_channels = block_channels

    def forward(self, x):
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        return x
