import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from exp.exp_basic import register_model


class Inception1DBlock(nn.Module):
    """
    Inception1D模块，用于处理一维光谱数据
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 9, 15], use_bottleneck=True, bottleneck_channels=32, activation=nn.ReLU()):
        super(Inception1DBlock, self).__init__()
        
        self.use_bottleneck = use_bottleneck
        self.bottlenecks = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.activation = activation
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        
        # 1x1卷积分支
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 其他卷积分支
        for k in kernel_sizes:
            if use_bottleneck:
                self.bottlenecks.append(nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1))
                self.convs.append(nn.Sequential(
                    nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_channels)
                ))
            else:
                self.convs.append(nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_channels)
                ))
        
    def forward(self, x):
        # 1x1卷积分支
        conv1x1_output = self.activation(self.conv1x1(x))
        
        # 其他卷积分支
        conv_outputs = [conv1x1_output]
        for i, conv in enumerate(self.convs):
            if self.use_bottleneck:
                bottleneck_output = self.bottlenecks[i](x)
                conv_outputs.append(self.activation(conv(bottleneck_output)))
            else:
                conv_outputs.append(self.activation(conv(x)))
        
        # 最大池化分支
        maxpool_output = self.activation(self.maxpool_conv(x))
        conv_outputs.append(maxpool_output)
        
        # 连接所有分支
        return torch.cat(conv_outputs, dim=1)


class InceptionModule(nn.Module):
    """
    包含多个Inception1D块的模块
    """
    def __init__(self, in_channels, config_list):
        """
        Args:
            in_channels: 输入通道数
            config_list: 列表，每个元素是(out_channels, kernel_sizes, bottleneck_channels)的元组
        """
        super(InceptionModule, self).__init__()
        
        self.inception_blocks = nn.ModuleList()
        current_channels = in_channels
        
        for out_channels, kernel_sizes, bottleneck_channels in config_list:
            inception_block = Inception1DBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_sizes=kernel_sizes,
                use_bottleneck=True,
                bottleneck_channels=bottleneck_channels
            )
            self.inception_blocks.append(inception_block)
            # 更新当前通道数 (每个分支的输出通道数 + 最大池化分支)
            current_channels = out_channels * (len(kernel_sizes) + 2)
        
        self.output_channels = current_channels
    
    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation模块，用于增强重要特征
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y


@register_model('Inception1DModel')
class Inception1DModel(nn.Module):
    """
    基于Inception1D的恒星光谱数据分析模型
    """
    def __init__(self, configs):
        super(Inception1DModel, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.feature_size
        self.pred_len = configs.label_size
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception模块配置
        inception_configs = [
            # (out_channels, kernel_sizes, bottleneck_channels)
            (32, [3, 5, 7], 16),
            (32, [3, 5, 7], 16),
            (64, [3, 5, 7, 11], 16),
            (64, [3, 5, 7, 11], 32),
            (128, [3, 5, 7, 11, 15], 32)
        ]
        
        # Inception模块
        self.inception_module = InceptionModule(32, inception_configs)
        
        # Squeeze-and-Excitation模块
        self.se_module = SqueezeExcitation(self.inception_module.output_channels)
        
        # 计算经过初始卷积和池化后的序列长度
        self.feature_len = self.seq_len // 4  # 两次步长为2的操作
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 全连接层
        fc_input_size = self.inception_module.output_channels * 2  # 平均池化和最大池化的结果拼接
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(64, self.pred_len)
        )
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'regression':
            return self.regression(x_enc)
        return None
    
    def regression(self, x):
        """
        执行回归任务
        x: [batch_size, seq_len, enc_in]
        返回: [batch_size, pred_len]
        """
        batch_size = x.size(0)
        x=x.unsqueeze(1) #[batch_size,1,seq_len]
        # 转换输入形状为 [batch_size, channels, seq_len]
        #x = x.permute(0, 2, 1)
        
        # 初始卷积层
        x = self.initial_conv(x)
        
        # Inception模块
        x = self.inception_module(x)
        
        # Squeeze-and-Excitation模块
        x = self.se_module(x)
        
        # 全局池化
        avg_pooled = self.global_avg_pool(x).view(batch_size, -1)
        max_pooled = self.global_max_pool(x).view(batch_size, -1)
        
        # 拼接池化结果
        x = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # 全连接层
        x = self.fc_layers(x)
        
        return x 