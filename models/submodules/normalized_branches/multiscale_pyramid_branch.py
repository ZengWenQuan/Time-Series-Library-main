import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_normalized_branch

class MultiScaleConvBlock(nn.Module):
    """
    多尺度卷积块: 使用多个不同尺寸的卷积核并行提取特征。
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2)
            for k in kernel_sizes
        ])
        self.fusion_conv = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        multi_scale_features = [conv(x) for conv in self.convs]
        concatenated_features = torch.cat(multi_scale_features, dim=1)
        fused_features = self.fusion_conv(concatenated_features)
        return F.relu(self.bn(fused_features))

@register_normalized_branch
class MultiScalePyramidBranch(nn.Module):
    """
    多尺度金字塔分支 (v3)。
    在卷积金字塔后增加自适应池化层，以输出固定长度的特征。
    """
    def __init__(self, config):
        super(MultiScalePyramidBranch, self).__init__()
        
        # --- 1. 构建网络层 --- 
        pyramid_layers = []
        in_channels = config.get('in_channels', 1)
        
        for layer_config in config['pyramid_layers']:
            out_channels = layer_config['out_channels']
            kernel_sizes = layer_config.get('kernel_sizes', [3, 5, 7])
            
            pyramid_layers.append(MultiScaleConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes
            ))
            
            pool_size = layer_config.get('pool_size')
            if pool_size and pool_size > 1:
                pyramid_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = layer_config['out_channels']

        self.pyramid = nn.Sequential(*pyramid_layers)
        
        # --- 2. 新增：自适应池化层 --- 
        self.target_output_length = config.get('target_output_length')
        if self.target_output_length and self.target_output_length > 0:
            self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_output_length)
        else:
            self.adaptive_pool = nn.Identity() # 如果不指定，则不进行池化

        # --- 3. 保存输出维度信息 --- 
        self.output_channels = in_channels
        # 如果不进行自适应池化，需要用公式计算长度
        if not isinstance(self.adaptive_pool, nn.AdaptiveAvgPool1d):
            L_in = config['input_len']
            for layer_cfg in config['pyramid_layers']:
                L_in = L_in // layer_cfg.get('pool_size', 1)
            self.output_length = L_in
        else:
            self.output_length = self.target_output_length

        self.output_dim = self.output_channels * self.output_length
        self.output_shape_tuple = (self.output_channels, self.output_length)

    def forward(self, x):
        # 保证输入是 (B, C, L) 的形状
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pyramid(x)
        x = self.adaptive_pool(x)
        return x