import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class InceptionBlock(nn.Module):
    """
    用于处理光谱数据的Inception模块
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 11], dilations=[1, 1, 1, 1]):
        super(InceptionBlock, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.conv_list = nn.ModuleList()
        
        # 为每个kernel size创建一个卷积层
        for k, d in zip(kernel_sizes, dilations):
            # 计算padding使得输出维度保持不变
            padding = (d * (k - 1)) // 2
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=padding, dilation=d)
            self.conv_list.append(conv)
        
        # 1x1卷积进行降维
        self.bottleneck = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        
    def forward(self, x):
        # 输入x的形状应为 [batch_size, channels, seq_len]
        conv_outputs = []
        
        for conv in self.conv_list:
            conv_outputs.append(conv(x))
        
        # 沿通道维度连接所有卷积结果
        x = torch.cat(conv_outputs, dim=1)
        
        # 通过瓶颈层
        x = self.bottleneck(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块，用于处理光谱数据的patch
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
        
        # 线性变换
        q = self.wq(x)  # [batch_size, seq_len, d_model]
        k = self.wk(x)  # [batch_size, seq_len, d_model]
        v = self.wv(x)  # [batch_size, seq_len, d_model]
        
        # 重塑以得到多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力权重
        energy = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        # 将注意力权重应用于值
        x = torch.matmul(attention, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 合并多头
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # 最后的线性层
        x = self.fc(x)  # [batch_size, seq_len, d_model]
        
        return x


class PatchProcessor(nn.Module):
    """
    处理光谱数据的patch，应用多头自注意力
    """
    def __init__(self, patch_size, d_model, num_heads, dim_feedforward=1024, dropout=0.1):
        super(PatchProcessor, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # patch到嵌入的线性变换
        self.patch_to_embedding = nn.Linear(patch_size, d_model)
        
        # 自注意力模块
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x的形状应为 [batch_size, num_patches, d_model]
        # 注意：x已经是嵌入后的形式，不需要再次变换
        
        # 应用自注意力
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Model(nn.Module):
    """
    用于恒星光谱数据的新模型，实现基于patch的处理和inception网络
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = 64
        self.overlap = 18
        self.stride = self.patch_size - self.overlap  # 46
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        
        # 计算需要的patch数量，限制最大数量为100
        self.num_patches = min(100, math.ceil((self.seq_len - self.patch_size) / self.stride) + 1)
        
        # 计算嵌入维度
        self.d_model = configs.d_model  # 使用命令行参数中的值
        
        # patch到嵌入的线性变换
        self.patch_to_embedding = nn.Linear(self.patch_size, self.d_model)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        
        # Patch处理器
        self.patch_processor = PatchProcessor(
            patch_size=self.patch_size, 
            d_model=self.d_model, 
            num_heads=configs.n_heads, 
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout
        )
        
        # 将处理后的patch映射回原始维度的卷积层
        self.patch_to_feature_map = nn.ConvTranspose1d(
            in_channels=self.d_model, 
            out_channels=1, 
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=(self.patch_size - self.stride) // 2
        )
        
        # Inception网络
        inception_layers = []
        channels = [1, 8, 8]
        for i in range(len(channels) - 1):
            inception_layers.append(
                InceptionBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_sizes=[3, 5],
                    dilations=[1, 1]
                )
            )
            inception_layers.append(nn.ReLU())
            inception_layers.append(nn.BatchNorm1d(channels[i + 1]))
        
        self.inception_network = nn.Sequential(*inception_layers)
        
        # 用于将inception结果映射到预测结果的层
        # 计算inception网络输出的特征长度
        self.feature_len = self.seq_len  # 卷积后保持原始长度
        
        # 自适应池化层，将不同长度的特征映射到固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(32)
        
        # 全连接层，用于回归任务
        self.fc1 = nn.Linear(32 * channels[-1], 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc_out = nn.Linear(16, self.pred_len)
        
        # 激活和dropout
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(configs.dropout)
        
    def _create_patches(self, x):
        """
        将输入序列切分为重叠的patch
        x: [batch_size, seq_len, channels]
        返回: [batch_size, num_patches, patch_size]
        """
        batch_size, seq_len, channels = x.size()
        device = x.device
        
        # 创建用于存储patch的张量
        patches = torch.zeros(batch_size, self.num_patches, self.patch_size, device=device)
        
        # 切分序列为patch，限制最大数量
        for i in range(min(self.num_patches, math.ceil((seq_len - self.patch_size) / self.stride) + 1)):
            # 计算当前patch的起始位置
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_size
            
            # 处理超出序列长度的情况
            if end_idx <= seq_len:
                patches[:, i, :] = x[:, start_idx:end_idx, 0]
            else:
                # 计算需要填充的长度
                padding_len = end_idx - seq_len
                valid_len = self.patch_size - padding_len
                
                if valid_len > 0:  # 确保有有效数据可以复制
                    # 将有效部分复制到patch中
                    patches[:, i, :valid_len] = x[:, start_idx:seq_len, 0]
                    
                    # 使用最后的数据重复填充剩余部分
                    patches[:, i, valid_len:] = x[:, -1, 0].unsqueeze(1).repeat(1, padding_len)
                else:
                    # 如果没有有效数据，全部使用最后一个点填充
                    patches[:, i, :] = x[:, -1, 0].unsqueeze(1).repeat(1, self.patch_size)
        
        return patches
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'regression':
            return self.regression(x_enc)
        return None
    
    def regression(self, x_enc):
        """
        执行回归任务
        x_enc: [batch_size, seq_len, enc_in]
        返回: [batch_size, pred_len]
        """
        # 切分为patch
        patches = self._create_patches(x_enc)  # [batch_size, num_patches, patch_size]
        batch_size = patches.size(0)
        
        # 将patch转换为嵌入
        patches_embed = self.patch_to_embedding(patches)  # [batch_size, num_patches, d_model]
        
        # 添加位置编码
        patches_embed = patches_embed + self.position_encoding
        
        # 使用patch处理器处理嵌入
        processed_patches = self.patch_processor(patches_embed)  # [batch_size, num_patches, d_model]
        
        # 将处理后的patch重塑为适合ConvTranspose1d的形状
        processed_patches = processed_patches.permute(0, 2, 1)  # [batch_size, d_model, num_patches]
        
        # 将处理后的patch映射回原始维度
        feature_map = self.patch_to_feature_map(processed_patches)  # [batch_size, 1, seq_len]
        
        # 应用Inception网络
        inception_output = self.inception_network(feature_map)  # [batch_size, channels, seq_len]
        
        # 自适应池化以得到固定长度的特征
        pooled_output = self.adaptive_pool(inception_output)  # [batch_size, channels, 32]
        
        # 展平
        flattened = pooled_output.view(batch_size, -1)  # [batch_size, channels*32]
        
        # 全连接层
        x = self.act(self.fc1(flattened))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        output = self.fc_out(x)
        
        return output 