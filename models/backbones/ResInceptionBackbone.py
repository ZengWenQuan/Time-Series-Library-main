# /models/backbones/ResInceptionBackbone.py (Calculates output dims)

import torch
import torch.nn as nn
from models.registries import register_backbone

class _MultiScaleInceptionModule(nn.Module):
    """多尺度Inception模块，作为残差块的核心 (采用精确通道分配)"""
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, use_batch_norm):
        super().__init__()
        
        num_branches = len(kernel_sizes)
        if out_channels < num_branches:
            raise ValueError(
                f"out_channels ({out_channels}) must be at least the number of branches ({num_branches})"
            )

        # --- 精确通道分配方案 ---
        self.branches = nn.ModuleList()
        
        # 前 n-1 个分支
        branch_channels = out_channels // num_branches
        for i in range(num_branches - 1):
            self.branches.append(
                nn.Conv1d(in_channels, branch_channels, 
                          kernel_size=kernel_sizes[i], stride=stride, padding=kernel_sizes[i] // 2)
            )
        
        # 最后一个分支，负责补足剩余的通道
        last_branch_channels = out_channels - (num_branches - 1) * branch_channels
        self.branches.append(
            nn.Conv1d(in_channels, last_branch_channels, 
                      kernel_size=kernel_sizes[-1], stride=stride, padding=kernel_sizes[-1] // 2)
        )

        # 1x1卷积，用于学习如何融合不同尺度的特征
        # 此时输入通道数之和严格等于out_channels
        self.fusion_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x_concat = torch.cat(branch_outputs, dim=1)
        x_fused = self.fusion_conv(x_concat)
        x_fused = self.bn(x_fused)
        return x_fused

class _ResInceptionBlock(nn.Module):
    """残差-Inception块 (修正后)"""
    def __init__(self, in_channels, out_channels, stride, kernel_sizes, use_batch_norm):
        super().__init__()
        
        main_path_layers = [
            nn.BatchNorm1d(in_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            _MultiScaleInceptionModule(in_channels, out_channels, kernel_sizes, stride=stride, use_batch_norm=use_batch_norm),
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        ]
        self.main_path = nn.Sequential(*main_path_layers)

        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            if use_batch_norm:
                shortcut_layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

        self.final_bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.final_relu = nn.ReLU(inplace=False)

    def forward(self, x):
        main_out = self.main_path(x)
        shortcut_out = self.shortcut(x)
        
        return self.final_relu(self.final_bn(main_out) + shortcut_out)

@register_backbone
class ResInceptionBackbone(nn.Module):
    """融合ResNet和Inception思想的一维骨干网络"""
    def __init__(self, cfg):
        super().__init__()
        
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        stem_cfg = cfg['stem']
        block_configs = cfg['blocks']
        use_adaptive_pool = cfg['use_adaptive_pool']
        use_batch_norm = cfg['use_batch_norm']

        # --- 动态计算输出长度 --- 
        current_length = input_length

        stem_layers = [
            nn.Conv1d(in_channels, stem_cfg['out_channels'], 
                      kernel_size=stem_cfg['kernel_size'], 
                      stride=stem_cfg['stride'], 
                      padding=stem_cfg['kernel_size'] // 2, bias=False)
        ]
        current_length = self._calculate_output_length(current_length, stem_cfg['kernel_size'], stem_cfg['stride'], stem_cfg['kernel_size'] // 2)
        
        if use_batch_norm:
            stem_layers.append(nn.BatchNorm1d(stem_cfg['out_channels']))
        stem_layers.append(nn.ReLU(inplace=False))
        self.stem = nn.Sequential(*stem_layers)

        self.blocks = nn.ModuleList()
        current_channels = stem_cfg['out_channels']

        for block_cfg in block_configs:
            out_channels = block_cfg['out_channels']
            stride = block_cfg['stride']
            num_repeats = block_cfg['num_repeats']
            kernel_sizes = block_cfg['inception_kernel_sizes']

            self.blocks.append(_ResInceptionBlock(current_channels, out_channels, stride, kernel_sizes, use_batch_norm))
            if stride > 1:
                k = kernel_sizes[0]
                p = k // 2
                current_length = self._calculate_output_length(current_length, k, stride, p)
            
            current_channels = out_channels
            for _ in range(num_repeats - 1):
                self.blocks.append(_ResInceptionBlock(current_channels, out_channels, 1, kernel_sizes, use_batch_norm))

        if use_adaptive_pool:
            self.output_length = 1
        else:
            self.output_length = current_length
            
        self.output_channels = current_channels

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        """根据PyTorch Conv1d文档的公式计算输出长度"""
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        #x = self.final_pool(x)
        return x