import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_local_branch

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention module."""
    def __init__(self, num_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积块: 使用多个不同尺寸的卷积核并行提取特征。
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], use_batch_norm=True):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.fusion_conv = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.Identity()
        self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x):
        multi_scale_features = [conv(x) for conv in self.convs]
        concatenated_features = torch.cat(multi_scale_features, dim=1)
        fused_features = self.fusion_conv(concatenated_features)
        bn_features = self.bn(fused_features)
        attended_features = self.channel_attention(bn_features)
        return F.relu(attended_features)

@register_local_branch
class MultiScalePyramidBranch(nn.Module):
    """
    多尺度金字塔分支 (v3)。
    在卷积金字塔后增加自适应池化层，以输出固定长度的特征。
    """
    def __init__(self, config):
        super(MultiScalePyramidBranch, self).__init__()
        
        # --- 1. 获取全局设置 ---
        use_batch_norm = config.get('use_batch_norm', True)

        # --- 2. 构建网络层 --- 
        pyramid_layers = []
        in_channels = config.get('in_channels', 1)
        
        for layer_config in config['pyramid_layers']:
            out_channels = layer_config['out_channels']
            kernel_sizes = layer_config.get('kernel_sizes', [3, 5, 7])
            
            pyramid_layers.append(MultiScaleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                use_batch_norm=use_batch_norm # 传递全局BN开关
            ))
            
            pool_size = layer_config.get('pool_size')
            if pool_size and pool_size > 1:
                pyramid_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = layer_config['out_channels']

        self.pyramid = nn.Sequential(*pyramid_layers)
        
        # --- 3. 保存输出维度信息 --- 
        
        # 基于金字塔结构计算最终输出长度
        L_in = config['input_len']
        for layer_cfg in config['pyramid_layers']:
            L_in = L_in // layer_cfg.get('pool_size', 1)
        self.output_channels = in_channels
        self.output_length = L_in

    def forward(self, x):
        x = self.pyramid(x)
        return x