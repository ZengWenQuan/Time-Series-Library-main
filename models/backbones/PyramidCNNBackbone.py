
import torch
import torch.nn as nn
from models.registries import register_backbone

class _BasicBlock(nn.Module):
    """一个基础的1D残差块 (Conv-BN-ReLU -> Conv-BN)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_batch_norm):
        super().__init__()

        padding = kernel_size // 2

        self.main_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            self.shortcut = nn.Sequential(*layers)
        
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut_out = self.shortcut(x)
        main_out = self.main_path(x)
        return self.final_relu(main_out + shortcut_out)

@register_backbone
class PyramidCNNBackbone(nn.Module):
    """金字塔卷积网络主干，用于多尺度特征提取和下采样"""
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        stem_cfg = cfg['stem']
        stages_cfg = cfg['stages']
        use_batch_norm = cfg.get('use_batch_norm', True)

        current_length = input_length

        # --- Stem层 (初始特征提取和下采样) ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_cfg['out_channels'], kernel_size=stem_cfg['kernel_size'], stride=stem_cfg['stride'], padding=stem_cfg['padding'], bias=False),
            nn.BatchNorm1d(stem_cfg['out_channels']) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        current_channels = stem_cfg['out_channels']
        current_length = self._calculate_output_length(current_length, stem_cfg['kernel_size'], stem_cfg['stride'], stem_cfg['padding'])

        # --- 构建多个金字塔阶段 ---
        self.stages = nn.ModuleList()
        for stage_cfg in stages_cfg:
            stage_layers = []
            num_blocks = stage_cfg['num_blocks']
            out_channels = stage_cfg['out_channels']
            kernel_size = stage_cfg['kernel_size']
            stride = stage_cfg['stride']

            # 每个阶段的第一个块负责下采样
            stage_layers.append(_BasicBlock(current_channels, out_channels, kernel_size, stride, use_batch_norm))
            current_channels = out_channels
            current_length = self._calculate_output_length(current_length, kernel_size, stride, kernel_size // 2)

            # 该阶段的其余块
            for _ in range(num_blocks - 1):
                stage_layers.append(_BasicBlock(current_channels, out_channels, kernel_size, 1, use_batch_norm))
            
            self.stages.append(nn.Sequential(*stage_layers))

        self.output_channels = current_channels
        self.output_length = current_length

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x
