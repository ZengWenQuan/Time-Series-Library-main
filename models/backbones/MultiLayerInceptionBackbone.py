# /home/irving/workspace/Time-Series-Library-main/models/backbones/MultiLayerInceptionBackbone.py

import torch
import torch.nn as nn
from models.registries import register_backbone, BLOCKS

class ResidualBlock(nn.Module):
    """残差连接块"""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

@register_backbone
class MultiLayerInceptionBackbone(nn.Module):
    """
    多层多尺度Inception网络，专门用于光谱特征提取
    Multi-layer Multi-scale Inception Network for spectral feature extraction
    """
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layer_configs = cfg['layers']

        # 计算输出维度用于后续模块
        self.output_channels = self.layer_configs[-1]['out_channels']
        # 输出长度假设不变（使用same padding）
        self.output_length = cfg['input_len']

        # 构建多层inception网络
        for i, layer_cfg in enumerate(self.layer_configs):
            # 第一层的输入通道数是原始输入
            if i == 0:
                layer_cfg['in_channels'] = cfg['input_channels']
            else:
                # 后续层的输入通道数是前一层的输出通道数
                layer_cfg['in_channels'] = self.layer_configs[i-1]['out_channels']

            # 使用注册的MultiScaleInception模块
            inception_layer = BLOCKS[layer_cfg['name']](layer_cfg)

            # 添加残差连接（当输入输出维度匹配时）
            if i > 0 and layer_cfg['in_channels'] == layer_cfg['out_channels']:
                # 包装为残差块
                residual_layer = ResidualBlock(inception_layer)
                self.layers.append(residual_layer)
            else:
                self.layers.append(inception_layer)

            # 可选的池化层用于降维
            if layer_cfg.get('pooling', False):
                pool_layer = nn.MaxPool1d(
                    kernel_size=layer_cfg['pooling']['kernel_size'],
                    stride=layer_cfg['pooling']['stride'],
                    padding=layer_cfg['pooling'].get('padding', 0)
                )
                self.layers.append(pool_layer)
                # 更新输出长度
                self.output_length = self._calculate_conv_output_length(
                    self.output_length,
                    layer_cfg['pooling']['kernel_size'],
                    layer_cfg['pooling']['stride'],
                    layer_cfg['pooling'].get('padding', 0)
                )

        # 全局特征融合层
        if cfg.get('global_fusion', True):
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.global_fc = nn.Linear(self.output_channels, self.output_channels // 4)
            self.global_activation = nn.ReLU()
            self.channel_attention = nn.Linear(self.output_channels // 4, self.output_channels)
            self.sigmoid = nn.Sigmoid()

    def _calculate_conv_output_length(self, input_length, kernel_size, stride, padding):
        """计算卷积后的序列长度"""
        return (input_length + 2 * padding - kernel_size) // stride + 1

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, channels, length] 输入光谱数据
        Returns:
            torch.Tensor: [batch_size, output_channels, output_length] 提取的多尺度特征
        """
        features = x

        # 通过多层inception网络
        for layer in self.layers:
            features = layer(features)

        # 可选的通道注意力机制
        if hasattr(self, 'global_pool'):
            # 计算通道注意力权重
            global_feat = self.global_pool(features).squeeze(-1)  # [B, C]
            attention_weights = self.sigmoid(
                self.channel_attention(
                    self.global_activation(self.global_fc(global_feat))
                )
            ).unsqueeze(-1)  # [B, C, 1]

            # 应用注意力权重
            features = features * attention_weights

        return features