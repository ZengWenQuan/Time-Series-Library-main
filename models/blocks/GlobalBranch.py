# /home/irving/workspace/Time-Series-Library-main/models/blocks/GlobalBranch.py

import torch
import torch.nn as nn
from models.registries import register_global_branch

class MultiLayerCNN(nn.Module):
    """多层CNN模块"""
    def __init__(self, layer_configs, use_batch_norm=True, dropout_rate=0.1):
        super().__init__()
        layers = []

        for cfg in layer_configs:
            # 卷积层
            layers.append(nn.Conv1d(
                cfg['in_channels'], cfg['out_channels'], cfg['kernel_size'],
                stride=cfg.get('stride', 1), padding=cfg['kernel_size'] // 2
            ))
            # BatchNorm（可选）
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(cfg['out_channels']))
            # 激活和Dropout
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        x = self.layers(x)

        # 简单残差连接（如果输入输出维度匹配）
        if identity.shape == x.shape:
            x = x + identity

        return x

@register_global_branch
class GlobalBranch(nn.Module):
    """改进的全局分支：直接对时间序列应用自注意力 + 多层CNN"""

    def __init__(self, cfg):
        super().__init__()

        # 基础参数
        self.in_channels = cfg['in_channels']
        self.out_channels = cfg['cnn_layers'][-1]['out_channels']
        self.use_batch_norm =cfg['use_batch_norm']
        self.dropout_rate = cfg['dropout_rate']

        # 注意力配置
        attention_dim =cfg['in_channels']

        # 核心层 - 将通道维度投影到注意力维度
        #self.input_projection = nn.Linear(self.in_channels, attention_dim)
        self.attention = nn.MultiheadAttention(
            attention_dim, cfg['n_heads'],
            dropout=self.dropout_rate, batch_first=True
        )
        self.norm = nn.LayerNorm(attention_dim)
        self.output_projection = nn.Linear(attention_dim, self.out_channels)
        #print(cfg)
        # 位置编码（可选）
        if cfg['use_positional_encoding']:
            max_length = cfg['max_length']
            self.pos_encoding = nn.Parameter(torch.randn(1, max_length, attention_dim) * 0.02)

        # 多层CNN
        cnn_layers = cfg['cnn_layers']
        self.cnn = MultiLayerCNN(cnn_layers, self.use_batch_norm, self.dropout_rate)

        # 输出维度
        self.output_channels = self.out_channels

        # 静态计算输出长度
        self.output_length = cfg.get('in_len')
        if self.output_length is not None:
            L_out = self.output_length # Attention doesn't change length
            for layer_cfg in cnn_layers:
                padding = layer_cfg['kernel_size'] // 2
                stride = layer_cfg.get('stride', 1)
                kernel_size = layer_cfg['kernel_size']
                dilation = 1 # Assuming default
                L_out = (L_out + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            self.output_length = L_out

    def forward(self, x):
        """[B, C, L] -> [B, out_channels, final_length]"""
        # 保存输入用于残差连接
        identity = x  # [B, C, L]

        # 1. 转换为序列格式
        x = x.permute(0, 2, 1)  # [B, L, C]

        # 2. 投影到注意力维度
        #features = self.input_projection(x)  # [B, L, attention_dim]
        features=x
        # 3. 位置编码
        if hasattr(self, 'pos_encoding'):
            seq_len = min(features.shape[1], self.pos_encoding.shape[1])
            features[:, :seq_len, :] = features[:, :seq_len, :] + self.pos_encoding[:, :seq_len, :]

        # 4. 自注意力（已有残差连接）
        attn_out, _ = self.attention(features, features, features)
        features = self.norm(features + attn_out)

        # 5. 投影到输出维度
        features = self.output_projection(features)  # [B, L, out_channels]

        # 6. 转换为CNN格式
        cnn_input = features.permute(0, 2, 1)  # [B, out_channels, L]

        # 7. CNN处理
        output = self.cnn(cnn_input)

        # 8. 跨层残差连接（如果维度匹配）
        if identity.shape[1] == output.shape[1] and identity.shape[2] == output.shape[2]:
            output = output + identity

        return output