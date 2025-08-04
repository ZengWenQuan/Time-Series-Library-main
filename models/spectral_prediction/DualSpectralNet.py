#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DualSpectralNet: Dual-Branch Spectral Prediction Network

基于2024-2025年最新研究，设计的双分支光谱预测模型：
- 连续谱分支：CNN+Transformer架构，提取全局特征
- 吸收线分支：多尺度CNN架构，提取局部特征
- 特征融合模块：交叉注意力机制进行特征融合
- 多尺度下采样：渐进式下采样，避免特征丢失

Author: Assistant
Date: 2025-08-03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
import math
from exp.exp_basic import register_model


class PositionalEncoding(nn.Module):
    """位置编码模块，用于Transformer，增强数值稳定性"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 使用更小的基数，减少数值范围
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))
        # 添加数值稳定性检查
        div_term = torch.clamp(div_term, min=1e-6, max=1e6)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 缩小位置编码的幅度
        pe = pe * 0.1
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 添加数值稳定性检查
        pos_enc = self.pe[:x.size(0), :]
        result = x + pos_enc
        # 防止数值过大
        return torch.clamp(result, min=-10.0, max=10.0)


class ContinuumBranch(nn.Module):
    """
    连续谱分支：使用CNN+Transformer提取全局特征
    适合处理连续谱的全局变化趋势
    """
    def __init__(self, config):
        super(ContinuumBranch, self).__init__()
        
        cnn_config = config['continuum_branch']['cnn']
        trans_config = config['continuum_branch']['transformer']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.1)
        
        # 1. CNN特征提取层
        self.cnn_layers = nn.ModuleList()
        in_channels = 1
        for layer_config in cnn_config['layers']:
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=layer_config['out_channels'],
                kernel_size=layer_config['kernel_size'],
                stride=layer_config['stride'],
                padding=layer_config['padding'],
                bias=False  # 始终使用bias以提高数值稳定性
            )
            
            layers = [conv]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_config['out_channels']))
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            
            self.cnn_layers.append(nn.Sequential(*layers))
            in_channels = layer_config['out_channels']
        
        # 2. 特征映射到Transformer维度
        self.feature_projection = nn.Linear(in_channels, trans_config['d_model'])
        
        # 3. Transformer组件（添加数值稳定性）
        self.pos_encoder = PositionalEncoding(trans_config['d_model'])
        self.norm = nn.LayerNorm(trans_config['d_model'], eps=1e-6)  # 增加eps值
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_config['d_model'],
            nhead=trans_config['n_heads'],
            dim_feedforward=trans_config['ffn_dim'],
            dropout=max(0.05, trans_config['dropout']),  # 最小0.05的dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=trans_config['num_layers']
        )
        
        # 4. 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.unsqueeze(1)  # -> [batch_size, 1, seq_len]
        #print("连续谱特征shape",x.shape,x[0,0,:10])
        # CNN特征提取
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        #print("连续谱特征经过卷积后shape",x.shape,x[0,0,:10])
        
        # 准备Transformer输入: [batch_size, channels, seq_len] -> [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)
        x = self.feature_projection(x)  # -> [batch_size, seq_len, d_model]
        
        #print("连续谱特征经过特征映射后shape",x.shape,x[0,0,:10])
        # 位置编码和标准化
        x_pos = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.norm(x_pos)
        
        # Transformer处理
        x = self.transformer(x)  # -> [batch_size, seq_len, d_model]
        
        # 全局特征提取
        x = x.permute(0, 2, 1)  # -> [batch_size, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # -> [batch_size, d_model]
        
        return x


class MultiScaleBlock(nn.Module):
    """多尺度卷积块，用于提取不同尺度的局部特征"""
    def __init__(self, in_channels, out_channels, kernel_sizes, use_attention=False, use_batch_norm=True, dropout_rate=0.1):
        super(MultiScaleBlock, self).__init__()
        
        self.use_attention = use_attention
        
        # 多个不同尺度的卷积分支
        self.branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
            
            layers = [conv]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            branch = nn.Sequential(*layers)
            self.branches.append(branch)
        
        # 特征融合
        total_channels = out_channels * len(kernel_sizes)
        fusion_layers = [nn.Conv1d(total_channels, out_channels, kernel_size=1, bias=True)]
        if use_batch_norm:
            fusion_layers.append(nn.BatchNorm1d(out_channels))
        fusion_layers.extend([
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        self.fusion = nn.Sequential(*fusion_layers)
        
        # 残差连接
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            residual_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)]
            if use_batch_norm:
                residual_layers.append(nn.BatchNorm1d(out_channels))
            self.residual = nn.Sequential(*residual_layers)
        
        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(out_channels, max(1, out_channels // 8), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, out_channels // 8), out_channels, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        # 多尺度特征提取
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 特征拼接和融合
        concat_features = torch.cat(branch_outputs, dim=1)
        fused_features = self.fusion(concat_features)
        
        # 注意力机制
        if self.use_attention:
            attention_weights = self.attention(fused_features)
            fused_features = fused_features * attention_weights
        
        # 残差连接
        return F.relu(fused_features + self.residual(x))


class AbsorptionBranch(nn.Module):
    """
    吸收线分支：使用多尺度CNN提取局部特征
    适合处理吸收线的局部精细结构
    """
    def __init__(self, config):
        super(AbsorptionBranch, self).__init__()
        
        absorption_config = config['absorption_branch']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.1)
        
        # 多尺度卷积块序列
        self.multiscale_blocks = nn.ModuleList()
        in_channels = 1
        
        for i, block_config in enumerate(absorption_config['blocks']):
            use_attention = block_config.get('use_attention', False)
            block = MultiScaleBlock(
                in_channels=in_channels,
                out_channels=block_config['out_channels'],
                kernel_sizes=block_config['kernel_sizes'],
                use_attention=use_attention,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate
            )
            self.multiscale_blocks.append(block)
            
            # 添加渐进式下采样
            if block_config.get('downsample', False):
                self.multiscale_blocks.append(
                    nn.AvgPool1d(
                        kernel_size=block_config.get('pool_size', 2),
                        stride=block_config.get('pool_stride', 2)
                    )
                )
            
            in_channels = block_config['out_channels']
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = in_channels
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.unsqueeze(1)  # -> [batch_size, 1, seq_len]
        
        #print("吸收线特征shape",x.shape,x[0,0,:10])
        # 多尺度特征提取
        for block in self.multiscale_blocks:
            x = block(x)
        
        #print("吸收线多尺度提取后特征shape",x.shape,x[0,0,:10])
        # 全局特征
        x = self.global_pool(x).squeeze(-1)  # -> [batch_size, feature_dim]
        
        #print("吸收线全局池化后特征shape",x.shape,x[0,:10])
        return x


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力特征融合模块 (修复版)
    - 移除clamp和自定义scale
    - 在交叉注意力前添加LayerNorm以增强数值稳定性
    """
    def __init__(self, continuum_dim, absorption_dim, fusion_dim, num_heads=4, dropout_rate=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # 特征维度统一
        self.continuum_proj = nn.Linear(continuum_dim, fusion_dim)
        self.absorption_proj = nn.Linear(absorption_dim, fusion_dim)
        
        # 为每个分支的输入添加LayerNorm，这是稳定注意力的关键
        self.norm_cont = nn.LayerNorm(fusion_dim)
        self.norm_abs = nn.LayerNorm(fusion_dim)
        
        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
    def forward(self, continuum_features, absorption_features):
        # 特征投影
        cont_proj = self.continuum_proj(continuum_features).unsqueeze(1)  # -> [B, 1, D]
        abs_proj = self.absorption_proj(absorption_features).unsqueeze(1)  # -> [B, 1, D]

        # 在注意力操作前进行LayerNorm，这是比clamp更根本的解决方案
        norm_cont_proj = self.norm_cont(cont_proj)
        norm_abs_proj = self.norm_abs(abs_proj)
        
        # 交叉注意力：连续谱特征查询吸收线特征
        cont_attended, _ = self.cross_attention(norm_cont_proj, norm_abs_proj, norm_abs_proj)
        # 交叉注意力：吸收线特征查询连续谱特征
        abs_attended, _ = self.cross_attention(norm_abs_proj, norm_cont_proj, norm_cont_proj)
        
        # 特征融合
        fused_features = torch.cat([
            cont_attended.squeeze(1), 
            abs_attended.squeeze(1)
        ], dim=1)
        
        result = self.fusion_layer(fused_features)
        return result


@register_model('DualSpectralNet')
class DualSpectralNet(nn.Module):
    """
    DualSpectralNet: 双分支光谱预测网络
    
    - 连续谱分支：CNN+Transformer，提取全局特征
    - 吸收线分支：多尺度CNN，提取局部特征
    - 特征融合：交叉注意力机制
    """
    def __init__(self, configs):
        super(DualSpectralNet, self).__init__()
        
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        
        # 加载模型配置
        if hasattr(configs, 'model_conf') and configs.model_conf:
            with open(configs.model_conf, 'r') as f:
                model_config = yaml.safe_load(f)
        else:
            raise ValueError("DualSpectralNet requires a model configuration file")
        
        # 从配置文件中获取BatchNorm和Dropout设置
        self.use_batch_norm = model_config.get('use_batch_norm', True)
        self.dropout_rate = model_config.get('dropout_rate', 0.1)
        self.initialization_method = model_config.get('initialization_method', 'kaiming')
        
        # 将这些设置添加到配置中，供子模块使用
        model_config['use_batch_norm'] = self.use_batch_norm
        model_config['dropout_rate'] = self.dropout_rate
        
        # 两个分支
        self.continuum_branch = ContinuumBranch(model_config)
        self.absorption_branch = AbsorptionBranch(model_config)
        
        # 特征融合
        continuum_dim = model_config['continuum_branch']['transformer']['d_model']
        absorption_dim = self.absorption_branch.feature_dim
        fusion_dim = model_config['fusion']['hidden_dim']
        
        self.feature_fusion = CrossAttentionFusion(
            continuum_dim=continuum_dim,
            absorption_dim=absorption_dim,
            fusion_dim=fusion_dim,
            num_heads=model_config['fusion']['num_heads'],
            dropout_rate=self.dropout_rate
        )
        
        # 最终预测头
        self.prediction_head = self._build_prediction_head(
            fusion_dim, 
            model_config['prediction_head']
        )
        
        # 模型初始化
        self._initialize_weights()
        
        # 打印模型信息
        self._print_model_info()
        
    def _build_prediction_head(self, input_dim, config):
        """构建预测头"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in config['hidden_dims']:
                # Linear层
            layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
            
            # BatchNorm层（可选）
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数和Dropout
            layers.extend([
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate)
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, self.label_size))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        采用更保守、更稳健的均匀分布初始化，以防止在FP16下发生数值溢出。
        """
        print(f"正在使用安全的均匀分布初始化...")
        
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                # 从一个小的、有界的均匀分布中初始化权重
                init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        #print("=" * 60)
        #print("DualSpectralNet 模型信息:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  使用BatchNorm: {self.use_batch_norm}")
        print(f"  Dropout率: {self.dropout_rate}")
        print(f"  初始化方法: {self.initialization_method}")
        print(f"  特征维度: {self.feature_size}")
        print(f"  标签维度: {self.label_size}")
        #print("=" * 60)
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'spectral_prediction':
            return self.regression(x_enc)
        raise ValueError("任务类型不对")
        return None
    
    def regression(self, x_enc):
        # 输入数据检查：仅在最开始检查一次
        if torch.isnan(x_enc).any() or torch.isinf(x_enc).any():
            raise ValueError("Input data to the model contains NaN or Inf values.")

        # 分离连续谱和归一化谱
        # 假设前半部分是连续谱，后半部分是吸收线谱
        continuum_spec = x_enc[:, :self.feature_size]
        absorption_spec = x_enc[:, self.feature_size:]
        #print("连续谱数据",continuum_spec[0,:10])
        #print("归一化谱数据",absorption_spec[0,:10])
        
        # 两个分支特征提取
        continuum_features = self.continuum_branch(continuum_spec)
        absorption_features = self.absorption_branch(absorption_spec)
        
        #print("连续谱分支数据",continuum_features[0,:10])
        #print("归一化谱分支数据",absorption_features[0,:10])
        # 特征融合
        fused_features = self.feature_fusion(continuum_features, absorption_features)
        
        #print("融合特征",fused_features[0,:10])
        # 最终预测
        output = self.prediction_head(fused_features)
        
        return output