# /models/backbones/ResNet50Backbone.py

import torch
import torch.nn as nn
from models.registries import register_backbone

class _BottleneckBlock(nn.Module):
    """ResNet-50使用的一维瓶颈残差块"""
    def __init__(self, in_channels, bottleneck_channels, out_channels, stride, use_batch_norm):
        super().__init__()

        # 主路径
        self.main_path = nn.Sequential(
            # 1x1卷积，降维
            nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            # 3x3卷积，特征提取（可能带步长进行降采样）
            nn.Conv1d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(bottleneck_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            # 1x1卷积，升维
            nn.Conv1d(bottleneck_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
        )

        # 快捷连接路径
        self.shortcut = nn.Sequential()
        # 当维度不匹配（步长或通道数改变）时，使用1x1卷积进行投影
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            if use_batch_norm:
                shortcut_layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut_out = self.shortcut(x)
        main_out = self.main_path(x)
        return self.final_relu(main_out + shortcut_out)

@register_backbone
class ResNet50Backbone1D(nn.Module):
    """一维版本的ResNet-50"""
    def __init__(self, cfg):
        super().__init__()

        # --- 解析配置 ---
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        stem_cfg = cfg['stem']
        block_configs = cfg['blocks']
        use_adaptive_pool = cfg['use_adaptive_pool']
        use_batch_norm = cfg['use_batch_norm']

        current_length = input_length

        # --- Stem层 ---
        self.stem_conv = nn.Conv1d(in_channels, stem_cfg['out_channels'], 
                                   kernel_size=stem_cfg['kernel_size'], 
                                   stride=stem_cfg['stride'], 
                                   padding=stem_cfg['kernel_size'] // 2, bias=False)
        current_length = self._calculate_output_length(current_length, stem_cfg['kernel_size'], stem_cfg['stride'], stem_cfg['kernel_size'] // 2)
        
        self.stem_bn = nn.BatchNorm1d(stem_cfg['out_channels']) if use_batch_norm else nn.Identity()
        self.stem_relu = nn.ReLU()
        self.stem_pool = nn.MaxPool1d(kernel_size=stem_cfg['pool_kernel_size'], 
                                      stride=stem_cfg['pool_stride'], 
                                      padding=stem_cfg['pool_padding'])
        current_length = self._calculate_output_length(current_length, stem_cfg['pool_kernel_size'], stem_cfg['pool_stride'], stem_cfg['pool_padding'])

        # --- 构建ResNet的4个阶段 ---
        self.blocks = nn.ModuleList()
        current_channels = stem_cfg['out_channels']

        for block_cfg in block_configs:
            out_channels = block_cfg['out_channels']
            bottleneck_channels = block_cfg['bottleneck_channels']
            stride = block_cfg['stride']
            num_repeats = block_cfg['num_repeats']

            # 每个阶段的第一个块使用指定的stride和变化的通道数
            self.blocks.append(_BottleneckBlock(current_channels, bottleneck_channels, out_channels, stride, use_batch_norm))
            current_channels = out_channels
            if stride > 1:
                # 降采样发生在3x3卷积上
                current_length = self._calculate_output_length(current_length, 3, stride, 1)

            # 该阶段后续的块
            for _ in range(num_repeats - 1):
                self.blocks.append(_BottleneckBlock(current_channels, bottleneck_channels, out_channels, 1, use_batch_norm))

        # --- 最终处理 ---
        if use_adaptive_pool:
            self.final_pool = nn.AdaptiveAvgPool1d(1)
            self.output_length = 1
        else:
            self.final_pool = nn.Identity()
            self.output_length = current_length
            
        self.output_channels = current_channels

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        for block in self.blocks:
            x = block(x)
            
        x = self.final_pool(x)
        return x
