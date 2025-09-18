# /home/irving/workspace/Time-Series-Library-main/models/blocks/MultiScaleInception.py

import torch
import torch.nn as nn
from models.registries import register_block
from models.submodules.attention import SEBlock

@register_block
class MultiScaleInception(nn.Module):
    """
    多尺度Inception模块，带有残差连接和SE注意力
    Multi-scale Inception module with residual connection and SE attention
    """
    def __init__(self, cfg):
        super().__init__()
        in_channels, out_channels, kernel_sizes = cfg['in_channels'], cfg['out_channels'], cfg['kernel_sizes']

        # 全局控制参数
        self.use_batch_norm = getattr(cfg, 'use_batch_norm', True)
        self.dropout_rate = getattr(cfg, 'dropout_rate', 0.1)

        # 多尺度Inception网络
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, k, padding=(k-1)//2) for k in kernel_sizes])

        # 瓶颈层
        self.bottleneck = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, 1)

        # BatchNorm（全局控制）
        if self.use_batch_norm:
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()

        self.activation = nn.ReLU()

        # Dropout（全局控制）
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        # SE注意力机制
        self.se_block = SEBlock(out_channels)

        # 残差连接的投影层（如果输入输出维度不匹配）
        self.residual_projection = None
        if in_channels != out_channels:
            self.residual_projection = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        # 多尺度特征提取
        features = [conv(x) for conv in self.convs]

        # 特征拼接和降维
        concat_features = torch.cat(features, dim=1)
        bottleneck_out = self.bottleneck(concat_features)

        # SE注意力
        se_out = self.se_block(bottleneck_out)

        # 残差连接
        if self.residual_projection is not None:
            residual = self.residual_projection(x)
        else:
            residual = x

        # 最终输出
        output = self.activation(self.norm(se_out + residual))
        return self.dropout(output)