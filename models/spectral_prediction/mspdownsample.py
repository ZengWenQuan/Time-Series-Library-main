#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MultiScale Pyramid Network for Spectral Analysis
多尺度金字塔网络用于光谱分析

Author: Assistant
Date: 2025-06-26

Architecture:
- 金字塔式多尺度特征提取
- 三分支并行处理：细粒度、中粒度、粗粒度
- 特征融合与注意力机制
- 简单高效的设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class PyramidBlock(nn.Module):
    """
    金字塔块：三分支并行处理不同尺度的特征
    
    Args:
        input_channel: 输入通道数
        output_channel: 输出通道数
        kernel_sizes: 三分支的卷积核大小列表
        use_batch_norm: 是否使用批归一化
        use_attention: 是否使用注意力机制
        attention_reduction: 注意力通道压缩比例
    """
    def __init__(self, input_channel, output_channel, kernel_sizes=[3, 5, 9], 
                 use_batch_norm=True, use_attention=True, attention_reduction=4):
        super(PyramidBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # 细粒度分支 - 小卷积核，保留细节
        self.fine_branch = self._make_branch(input_channel, output_channel, 
                                           kernel_sizes[0], use_batch_norm)
        
        # 中粒度分支 - 中等卷积核，平衡细节和上下文
        self.medium_branch = self._make_branch(input_channel, output_channel, 
                                             kernel_sizes[1], use_batch_norm)
        
        # 粗粒度分支 - 大卷积核，捕获全局上下文
        self.coarse_branch = self._make_branch(input_channel, output_channel, 
                                             kernel_sizes[2], use_batch_norm)
        
        # 残差连接
        self.residual = nn.Sequential()
        if input_channel != output_channel * 3:
            layers = [nn.Conv1d(input_channel, output_channel * 3, kernel_size=1, stride=2, bias=not use_batch_norm)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(output_channel * 3))
            self.residual = nn.Sequential(*layers)
            
        # 注意力机制
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(output_channel * 3, max(1, output_channel * 3 // attention_reduction), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, output_channel * 3 // attention_reduction), output_channel * 3, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_branch(self, input_channel, output_channel, kernel_size, use_batch_norm):
        """创建单个分支"""
        padding = kernel_size // 2
        layers = []
        mid_channel = (input_channel+output_channel)//2
        # 第一层卷积
        layers.append(nn.Conv1d(input_channel, mid_channel, kernel_size=kernel_size, 
                               padding=padding, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(mid_channel))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二层卷积
        layers.append(nn.Conv1d(mid_channel, mid_channel, kernel_size=kernel_size, 
                               padding=padding, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(mid_channel))
        layers.append(nn.ReLU(inplace=True))
        
        # 第三层卷积
        layers.append(nn.Conv1d(mid_channel, output_channel, kernel_size=kernel_size, stride=2,#stride=2下采样
                               padding=padding, bias=not use_batch_norm))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_channel))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 三个分支并行处理
        fine_out = self.fine_branch(x)
        medium_out = self.medium_branch(x)
        coarse_out = self.coarse_branch(x)
        
        # 拼接三个分支的输出
        pyramid_out = torch.cat([fine_out, medium_out, coarse_out], dim=1)
        
        # 注意力加权
        if self.use_attention:
            attention_weights = self.attention(pyramid_out)
            pyramid_out = pyramid_out * attention_weights
        
        # 残差连接
        residual_out = self.residual(x)
        
        return pyramid_out + residual_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class FeatureAggregator(nn.Module):
    """
    特征聚合器：将多尺度特征聚合为固定维度
    支持两种输入格式：
    1. 卷积特征图 [B, C, L] - 使用全局池化
    2. 序列特征 [B, L, C] - 使用平均池化
    """
    def __init__(self, input_channel, embedding_dim, use_batch_norm=True, input_type='conv'):
        super(FeatureAggregator, self).__init__()
        
        self.input_type = input_type  # 'conv' 或 'sequence'
        
        if input_type == 'conv':
            # 用于卷积特征图 [B, C, L]
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            layers = [nn.Conv1d(input_channel, embedding_dim, kernel_size=1, bias=not use_batch_norm)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(embedding_dim))
            layers.append(nn.ReLU(inplace=True))
            self.conv_proj = nn.Sequential(*layers)
        else:
            # 用于序列特征 [B, L, C]
            layers = [nn.Linear(input_channel, embedding_dim, bias=not use_batch_norm)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(embedding_dim))
            layers.append(nn.ReLU(inplace=True))
            self.linear_proj = nn.Sequential(*layers)
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        if self.input_type == 'conv':
            # 处理卷积特征图 [B, C, L]
            pooled = self.global_pool(x)  # [B, C, 1]
            embedded = self.conv_proj(pooled)  # [B, embedding_dim, 1]
            return embedded.squeeze(-1)  # [B, embedding_dim]
        else:
            # 处理序列特征 [B, L, C]
            # 在序列维度上平均池化
            pooled = x.mean(dim=1)  # [B, C]
            embedded = self.linear_proj(pooled)  # [B, embedding_dim]
            return embedded

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class MSPDownsample(nn.Module):
    """
    MultiScale Pyramid Network for Regression Tasks
    多尺度金字塔网络用于回归任务
    
    Args:
        config: 配置字典，包含以下参数
            - num_classes/label_size: 输出标签数量（回归任务）
            - pyramid_channels: 金字塔块的通道数列表
            - embedding_dim: 嵌入维度
            - dropout_rate: Dropout概率
            - fc_hidden_dims: 全连接层隐藏层维度列表
            - feature_size: 输入特征维度（固定）
    """
    def __init__(self, config):
        super(MSPDownsample, self).__init__()
        
        # 从配置中获取参数
        self.label_size = config.get('num_classes') or config.get('label_size') or 4
        self.pyramid_channels = config.get('pyramid_channels', [16, 32, 64])
        self.embedding_dim = config.get('embedding_dim', 128)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.fc_hidden_dims = config.get('fc_hidden_dims', [256, 128])
        
        # 新增配置参数
        self.kernel_sizes = config.get('kernel_sizes', [3, 5, 7])
        self.pool_size = config.get('pool_size', 2)
        self.use_batch_norm = config.get('batch_norm', True)
        self.use_attention = config.get('use_attention', True)
        self.attention_reduction = config.get('attention_reduction', 4)
        self.feature_size = config.get('feature_size')  # 输入特征维度
        
        # LSTM相关参数
        self.use_lstm = config.get('use_lstm', True)
        self.lstm_hidden_size = config.get('lstm_hidden_size', 64)
        self.lstm_num_layers = config.get('lstm_num_layers', 1)
        self.lstm_bidirectional = config.get('lstm_bidirectional', True)
        
        # 输入投影层
        layers = [nn.Conv1d(1, self.pyramid_channels[0], kernel_size=7, padding=3, bias=not self.use_batch_norm)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.pyramid_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*layers)
        
        # 金字塔块序列
        self.pyramid_blocks = nn.ModuleList()
        
        for i in range(len(self.pyramid_channels)):
            if i == 0:
                # 第一个块：从输入投影层的输出开始
                input_ch = self.pyramid_channels[0]
                output_ch = self.pyramid_channels[0]
            else:
                # 后续块：从前一个块的输出开始
                input_ch = self.pyramid_channels[i-1] * 3  # 三分支拼接
                output_ch = self.pyramid_channels[i]
            
            self.pyramid_blocks.append(nn.Sequential(
                PyramidBlock(input_ch, output_ch, self.kernel_sizes, 
                           self.use_batch_norm, self.use_attention, self.attention_reduction),
                nn.AvgPool1d(kernel_size=self.pool_size, stride=self.pool_size)  # 下采样
            ))
        
        # LSTM层（在金字塔块之后）
        if self.use_lstm:
            lstm_input_size = self.pyramid_channels[-1] * 3  # 最后一个金字塔块的输出通道数
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                bidirectional=self.lstm_bidirectional,
                dropout=self.dropout_rate if self.lstm_num_layers > 1 else 0
            )
            
            # 计算LSTM输出维度
            lstm_output_size = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)
            
            # 特征聚合器（输入为LSTM输出，序列格式）
            self.aggregator = FeatureAggregator(lstm_output_size, self.embedding_dim, self.use_batch_norm, input_type='sequence')
        else:
            # 不使用LSTM时，直接聚合金字塔特征（卷积格式）
            final_channels = self.pyramid_channels[-1] * 3  # 最后一个金字塔块的输出通道数
            self.aggregator = FeatureAggregator(final_channels, self.embedding_dim, self.use_batch_norm, input_type='conv')
        
        # 多层全连接网络（使用固定的nn.Linear层）
        self.fc_layers = nn.ModuleList()
        
        # 第一层：从嵌入维度到第一个隐藏层
        self.fc_layers.append(nn.Linear(self.embedding_dim, self.fc_hidden_dims[0]))
        if self.use_batch_norm:
            self.fc_layers.append(nn.BatchNorm1d(self.fc_hidden_dims[0]))
        self.fc_layers.append(nn.ReLU(inplace=True))
        self.fc_layers.append(nn.Dropout(self.dropout_rate))
        
        # 中间层
        for i in range(len(self.fc_hidden_dims) - 1):
            self.fc_layers.append(nn.Linear(self.fc_hidden_dims[i], self.fc_hidden_dims[i+1]))
            if self.use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(self.fc_hidden_dims[i+1]))
            self.fc_layers.append(nn.ReLU(inplace=True))
            self.fc_layers.append(nn.Dropout(self.dropout_rate))
        
        # 输出层
        self.output_layer = nn.Linear(self.fc_hidden_dims[-1], self.label_size)
        
        # 初始化权重（只需初始化顶层自定义模块）
        self._initialize_weights()
        
        print(f"初始化 MSPNet")
        print(f"输入特征维度: {self.feature_size}")
        print(f"金字塔通道数: {self.pyramid_channels}")
        print(f"卷积核大小: {self.kernel_sizes}")
        print(f"嵌入维度: {self.embedding_dim}")
        print(f"全连接层维度: {self.fc_hidden_dims}")
        print(f"批归一化: {self.use_batch_norm}")
        print(f"注意力机制: {self.use_attention}")
        if self.use_lstm:
            print(f"LSTM: 隐藏层{self.lstm_hidden_size}, 层数{self.lstm_num_layers}, 双向{self.lstm_bidirectional}")
        else:
            print(f"LSTM: 未启用")
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.orthogonal_(param.data)
                    else:
                        init.normal_(param.data)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] 1D input signal
        Returns:
            [batch_size, num_classes] regression output
        """
        # 确保输入形状正确
        if len(x.shape) == 2:
            B, L = x.size()
            # 检查输入维度是否匹配
            if self.feature_size is not None and L != self.feature_size:
                print(f"检测到新的输入维度: {L}")
                # 如果维度不匹配，可以选择填充或截断
                if L < self.feature_size:
                    # 填充
                    padding_size = self.feature_size - L
                    x = torch.cat([x, torch.zeros([B, padding_size]).to(x.device)], dim=1)
                else:
                    # 截断
                    x = x[:, :self.feature_size]
            
            # 填充使得长度能被处理（如果需要）
            # if L % 8 != 0:
            #     padding_size = 8 - (L % 8)
            #     x = torch.cat([x, torch.zeros([B, padding_size]).to(x.device)], dim=1)
            x = x.reshape(-1, 1, x.size(1))  # [batch_size, 1, seq_len]
        
        # 输入投影
        x = self.input_proj(x)
        
        # 通过金字塔块序列
        for pyramid_block in self.pyramid_blocks:
            x = pyramid_block(x)
        
        # LSTM处理（如果启用）
        if self.use_lstm:
            # 转换为序列格式：[B, C, L] -> [B, L, C]
            x = x.permute(0, 2, 1)  # [batch_size, seq_len, channels]
            
            # 通过LSTM
            lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_size * directions]
            
            # 特征聚合（序列格式输入）
            x = self.aggregator(lstm_out)  # [batch_size, embedding_dim]
        else:
            # 特征聚合（卷积格式输入）
            x = self.aggregator(x)  # [batch_size, embedding_dim]
        
        # 通过全连接层
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        # 输出层
        output = self.output_layer(x)
        
        return output
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'label_size': self.label_size,
            'feature_size': self.feature_size,
            'pyramid_channels': self.pyramid_channels,
            'embedding_dim': self.embedding_dim,
            'fc_hidden_dims': self.fc_hidden_dims,
            'dropout_rate': self.dropout_rate,
            'task_type': 'regression'
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print("=" * 50)
        print("MSPNet 多尺度金字塔网络信息:")
        print("=" * 50)
        print(f"总参数数量: {info['total_params']:,}")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"输入特征维度: {info['feature_size']}")
        print(f"输出标签数: {info['num_classes']}")
        print(f"金字塔通道数: {info['pyramid_channels']}")
        print(f"嵌入维度: {info['embedding_dim']}")
        print(f"全连接层维度: {info['fc_hidden_dims']}")
        print(f"Dropout率: {info['dropout_rate']}")
        print(f"任务类型: {info['task_type']}")
        print("=" * 50)


# 设置模型类，用于自动加载
MODEL_CLASS = MSPDownsample