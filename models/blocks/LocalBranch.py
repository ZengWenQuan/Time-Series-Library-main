# /home/irving/workspace/Time-Series-Library-main/models/blocks/LocalBranch.py

import torch.nn as nn
from models.registries import register_local_branch

class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

@register_local_branch
class LocalBranch(nn.Module):
    """
    多层CNN局部分支，用于提取局部特征
    Multi-layer CNN local branch for extracting local features
    """
    def __init__(self, cfg):
        super().__init__()

        # 全局控制参数
        self.use_batch_norm = getattr(cfg, 'use_batch_norm', True)
        self.dropout_rate = getattr(cfg, 'dropout_rate', 0.1)

        self.layers = nn.ModuleList()
        current_channels = cfg['in_channels']

        # 构建多层CNN网络
        layer_configs = cfg.get('layer_configs')
        print("查看下卷积层配置",layer_configs,"第一次卷积的通道数",current_channels)
        for layer_cfg in layer_configs:
            # 卷积层
            conv = nn.Conv1d(
                current_channels,
                layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                padding=layer_cfg['kernel_size'] // 2,
                stride=layer_cfg.get('stride', 1)
            )

            # 组装层
            layer_modules = [conv]

            # 批归一化（全局控制）
            if self.use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(layer_cfg['out_channels']))

            # 激活函数
            layer_modules.append(nn.ReLU())

            # Dropout（全局控制）
            if self.dropout_rate > 0:
                layer_modules.append(nn.Dropout(self.dropout_rate))

            layer_block = nn.Sequential(*layer_modules)

            # 添加残差连接（当输入输出维度匹配且步长为1时）
            if current_channels == layer_cfg['out_channels'] and layer_cfg.get('stride', 1) == 1:
                layer_block = ResidualBlock(layer_block)

            self.layers.append(layer_block)
            current_channels = layer_cfg['out_channels']

        # 输出维度
        self.output_channels = current_channels

        # 静态计算输出长度
        self.output_length = cfg.get('in_len')
        if self.output_length is not None:
            L_out = self.output_length
            for layer_cfg in layer_configs:
                padding = layer_cfg['kernel_size'] // 2
                stride = layer_cfg.get('stride', 1)
                kernel_size = layer_cfg['kernel_size']
                dilation = 1 # Assuming default
                L_out = (L_out + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            self.output_length = L_out

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, channels, length] 输入特征
        Returns:
            torch.Tensor: [batch_size, output_channels, output_length] 局部特征
        """
        for layer in self.layers:
            x = layer(x)

        return x