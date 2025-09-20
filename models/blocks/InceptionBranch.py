# /models/blocks/LocalBranch.py (With optional Channel Attention)

import torch
import torch.nn as nn
from models.registries import register_local_branch

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = channels // reduction_ratio
        if reduced_channels == 0:
            reduced_channels = 1
            
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C, L]
        b, c, _ = x.shape
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1)
        # Scale
        return x * y.expand_as(x)

class _MultiScaleBlock(nn.Module):
    """一个多尺度卷积块，模仿Inception的设计"""
    def __init__(self, in_channels, out_channels, kernel_sizes, stride, use_batch_norm, dropout_rate):
        super().__init__()

        num_branches = len(kernel_sizes)
        if out_channels < num_branches:
            raise ValueError(f"out_channels ({out_channels}) must be >= num_branches ({num_branches})")

        # --- 精确通道分配 ---
        self.branches = nn.ModuleList()
        branch_channels = out_channels // num_branches
        for i in range(num_branches - 1):
            self.branches.append(
                nn.Conv1d(in_channels, branch_channels, 
                          kernel_size=kernel_sizes[i], stride=stride, padding=kernel_sizes[i] // 2)
            )
        last_branch_channels = out_channels - (num_branches - 1) * branch_channels
        self.branches.append(
            nn.Conv1d(in_channels, last_branch_channels, 
                      kernel_size=kernel_sizes[-1], stride=stride, padding=kernel_sizes[-1] // 2)
        )

        # --- 特征融合 ---
        self.fusion_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        layers = []
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        self.post_fusion = nn.Sequential(*layers)

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        x_concat = torch.cat(branch_outputs, dim=1)
        x_fused = self.fusion_conv(x_concat)
        return self.post_fusion(x_fused)

@register_local_branch
class InceptionBranch(nn.Module):
    """
    多尺度CNN局部分支 (Inception-style) with optional Channel Attention
    """
    def __init__(self, cfg):
        super().__init__()

        # 全局控制参数
        use_batch_norm = cfg.get('use_batch_norm', True)
        dropout_rate = cfg.get('dropout_rate', 0.1)
        in_channels = cfg['in_channels']
        in_len = cfg['in_len']
        block_configs = cfg['blocks']

        self.blocks = nn.ModuleList()
        current_channels = in_channels
        current_length = in_len

        # 构建多层多尺度网络
        for block_cfg in block_configs:
            out_channels = block_cfg['out_channels']
            kernel_sizes = block_cfg['kernel_sizes']
            stride = block_cfg.get('stride', 1)

            block = _MultiScaleBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                stride=stride,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            self.blocks.append(block)
            current_channels = out_channels
            
            k = kernel_sizes[0]
            p = k // 2
            current_length = self._calculate_output_length(current_length, k, stride, p)

        self.output_channels = current_channels
        self.output_length = current_length

        # --- 可选的通道注意力机制 ---
        if cfg.get('use_channel_attention', False):
            self.channel_attention = SEBlock(
                channels=self.output_channels, 
                reduction_ratio=cfg.get('attention_reduction_ratio', 16)
            )
        else:
            self.channel_attention = nn.Identity()

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        """
        前向传播
        """
        for block in self.blocks:
            x = block(x)
        
        # 应用最终的通道注意力
        x = self.channel_attention(x)
        return x