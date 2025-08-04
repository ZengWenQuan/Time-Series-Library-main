import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AttentionBlock(nn.Module):
    """
    自注意力模块，用于捕获光谱特征之间的长距离依赖关系
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # 自注意力层
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + attended)
        
        # 前馈网络
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class SpectralConvBlock(nn.Module):
    """
    光谱卷积块，用于提取局部光谱特征
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(SpectralConvBlock, self).__init__()
        
        # 多尺度卷积层
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels // len(kernel_sizes), k, padding=k//2),
                    nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                    nn.ReLU()
                )
            )
            
        self.dropout = nn.Dropout(dropout)
        self.out_channels = out_channels
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, seq_len]
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(conv(x))
        
        # 合并多尺度特征
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    位置编码，用于为注意力层提供位置信息
    """
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1)]


@register_model('AttentionSpectrumNet')
class AttentionSpectrumNet(nn.Module):
    """
    AttentionSpectrumNet模型，结合了卷积网络和注意力机制，
    专为恒星光谱数据分析设计
    """
    def __init__(self, configs):
        super(AttentionSpectrumNet, self).__init__()
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size  # 光谱长度
        self.label_size = configs.label_size      # 输出标签数量
        
        # 模型配置
        self.embed_dim = configs.embed_dim if hasattr(configs, 'embed_dim') else 128
        self.num_heads = configs.num_heads if hasattr(configs, 'num_heads') else 4
        self.num_layers = configs.num_layers if hasattr(configs, 'num_layers') else 3
        self.conv_channels = configs.conv_channels if hasattr(configs, 'conv_channels') else [32, 64, 128]
        self.kernel_sizes = configs.kernel_sizes if hasattr(configs, 'kernel_sizes') else [3, 5, 7]
        self.patch_size = configs.patch_size if hasattr(configs, 'patch_size') else 64
        self.stride = configs.stride if hasattr(configs, 'stride') else 48  # 允许重叠
        self.dropout = configs.dropout_rate if hasattr(configs, 'dropout_rate') else 0.2
        self.reduction_factor = configs.reduction_factor if hasattr(configs, 'reduction_factor') else 4  # 特征降维因子
        
        # 特征提取阶段 - 卷积层
        self.conv_blocks = nn.ModuleList()
        in_channels = 1  # 初始输入通道为1
        
        for out_channels in self.conv_channels:
            self.conv_blocks.append(
                SpectralConvBlock(in_channels, out_channels, self.kernel_sizes, self.dropout)
            )
            in_channels = out_channels
        
        # 计算卷积处理后的特征长度
        self.conv_feature_length = self.feature_size // (2 ** len(self.conv_channels))
        self.conv_feature_dim = self.conv_channels[-1]
        
        # 降维层 - 减少后续注意力机制的计算复杂度
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.conv_feature_dim, self.embed_dim // self.reduction_factor),
            nn.ReLU(),
            nn.Linear(self.embed_dim // self.reduction_factor, self.embed_dim)
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        
        # 自注意力层
        self.attention_blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attention_blocks.append(
                AttentionBlock(self.embed_dim, self.num_heads, self.dropout)
            )
        
        # 分类头 - 使用注意力输出的平均池化结果
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # 回归头
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embed_dim // 2, self.label_size)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
                
    def extract_patches(self, x):
        """
        从光谱数据中提取重叠的补丁
        x: [batch_size, seq_len]
        """
        B, L = x.shape
        
        # 确定补丁数量
        num_patches = (L - self.patch_size) // self.stride + 1
        
        # 提取补丁
        patches = []
        for i in range(num_patches):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_size
            patch = x[:, start_idx:end_idx]
            patches.append(patch)
            
        # 如果最后一个补丁没有覆盖到结尾，添加最后一个补丁
        if end_idx < L:
            patches.append(x[:, -self.patch_size:])
            
        # 将补丁堆叠成一个张量 [batch_size, num_patches, patch_size]
        x_patches = torch.stack(patches, dim=1)
        
        return x_patches
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """符合项目架构的前向传播函数"""
        if self.task_name == 'regression' or self.task_name == 'stellar_parameter_estimation':
            return self.regression(x_enc)
        return None
    
    def regression(self, x):
        """
        执行回归任务
        x: [batch_size, seq_len]
        返回: [batch_size, label_size]
        """
        batch_size = x.shape[0]
        
        # 提取重叠的补丁
        patches = self.extract_patches(x)  # [batch_size, num_patches, patch_size]
        
        # 每个补丁进行卷积特征提取
        # 首先调整补丁形状以适应卷积操作
        num_patches = patches.shape[1]
        patches = patches.reshape(-1, 1, self.patch_size)  # [batch_size*num_patches, 1, patch_size]
        
        # 应用卷积块
        for conv_block in self.conv_blocks:
            patches = conv_block(patches)  # [batch_size*num_patches, channels, reduced_len]
        
        # 全局平均池化每个补丁的卷积特征
        patches = F.adaptive_avg_pool1d(patches, 1).squeeze(-1)  # [batch_size*num_patches, channels]
        
        # 调整形状以便于注意力处理
        patches = patches.reshape(batch_size, num_patches, -1)  # [batch_size, num_patches, channels]
        
        # 降维
        patches = self.dim_reduction(patches)  # [batch_size, num_patches, embed_dim]
        
        # 添加位置编码
        patches = self.positional_encoding(patches)  # [batch_size, num_patches, embed_dim]
        
        # 添加分类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # [batch_size, num_patches+1, embed_dim]
        
        # 应用自注意力层
        for attn_block in self.attention_blocks:
            x = attn_block(x)  # [batch_size, num_patches+1, embed_dim]
        
        # 使用分类令牌作为输出特征
        x = x[:, 0]  # [batch_size, embed_dim]
        x = self.norm(x)
        
        # 预测输出
        x = self.regression_head(x)  # [batch_size, label_size]
        
        return x 