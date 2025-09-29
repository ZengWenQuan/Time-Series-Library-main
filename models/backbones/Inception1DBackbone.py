
import torch
import torch.nn as nn
from models.registries import register_backbone

class _InceptionModule(nn.Module):
    """一维Inception模块"""
    def __init__(self, in_channels, ch1x1, ch3x3_re, ch3x3, ch5x5_re, ch5x5, pool_proj, use_batch_norm):
        super().__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm1d(ch1x1) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
        )

        # 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, ch3x3_re, kernel_size=1),
            nn.BatchNorm1d(ch3x3_re) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv1d(ch3x3_re, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch3x3) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
        )

        # 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, ch5x5_re, kernel_size=1),
            nn.BatchNorm1d(ch5x5_re) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.Conv1d(ch5x5_re, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(ch5x5) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
        )

        # Max pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
        )
        
        self.output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)

@register_backbone
class Inception1DBackbone(nn.Module):
    """一维Inception网络主干"""
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        stem_cfg = cfg['stem']
        inception_configs = cfg['inception_modules']
        use_batch_norm = cfg.get('use_batch_norm', True)

        current_length = input_length

        # --- Stem层 ---
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_cfg['out_channels'], kernel_size=stem_cfg['kernel_size'], stride=stem_cfg['stride'], padding=stem_cfg['padding']),
            nn.BatchNorm1d(stem_cfg['out_channels']) if use_batch_norm else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=stem_cfg['pool_kernel_size'], stride=stem_cfg['pool_stride'], padding=stem_cfg['pool_padding'])
        )
        current_length = self._calculate_output_length(current_length, stem_cfg['kernel_size'], stem_cfg['stride'], stem_cfg['padding'])
        current_length = self._calculate_output_length(current_length, stem_cfg['pool_kernel_size'], stem_cfg['pool_stride'], stem_cfg['pool_padding'])
        
        # --- 构建Inception模块 ---
        self.inception_blocks = nn.ModuleList()
        current_channels = stem_cfg['out_channels']
        for module_cfg in inception_configs:
            module = _InceptionModule(current_channels, **module_cfg, use_batch_norm=use_batch_norm)
            self.inception_blocks.append(module)
            current_channels = module.output_channels

        self.output_channels = current_channels
        self.output_length = current_length

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = self.stem(x)
        for block in self.inception_blocks:
            x = block(x)
        return x
