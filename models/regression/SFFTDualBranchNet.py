import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from exp.exp_basic import register_model


class SFFTFeatureExtractor(nn.Module):
    """
    SFFT特征提取模块
    """
    def __init__(self, feature_size):
        super(SFFTFeatureExtractor, self).__init__()
        self.feature_size = feature_size
        
    def forward(self, x):
        # x shape: (batch_size, feature_size)
        # 进行短时傅里叶变换
        # 使用torch.stft进行STFT变换
        stft_result = torch.stft(
            x, 
            n_fft=min(512, self.feature_size//4),  # FFT窗口大小
            hop_length=min(128, self.feature_size//8),  # 跳跃长度
            win_length=min(256, self.feature_size//6),  # 窗口长度
            return_complex=True
        )
        
        # 取幅度谱
        magnitude = torch.abs(stft_result)  # shape: (batch_size, freq_bins, time_frames)
        
        return magnitude


class FullConvBranch(nn.Module):
    """
    全卷积分支，包含5层卷积+平均池化
    """
    def __init__(self, in_channels, base_channels=64):
        super(FullConvBranch, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(5):
            # 卷积层
            conv_layer = nn.Sequential(
                nn.Conv2d(current_channels, base_channels * (2**i), 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * (2**i)),
                nn.ReLU(inplace=True),
                # 3x3平均池化下采样
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
            self.layers.append(conv_layer)
            current_channels = base_channels * (2**i)
            
        self.output_channels = current_channels
    
    def forward(self, x):
        # x shape: (batch_size, channels, freq_bins, time_frames)
        for layer in self.layers:
            x = layer(x)
        return x


class InceptionBranch(nn.Module):
    """
    Inception风格分支，每层使用不同大小的卷积核
    """
    def __init__(self, in_channels, base_channels=32):
        super(InceptionBranch, self).__init__()
        
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(5):
            inception_layer = InceptionLayer(
                current_channels, 
                base_channels * (2**i),
                kernel_sizes=[1, 3, 5, 7]
            )
            self.layers.append(inception_layer)
            # 每个Inception层输出通道数是输入的4倍（4个不同kernel size的分支）
            current_channels = base_channels * (2**i) * 4
            
        self.output_channels = current_channels
    
    def forward(self, x):
        # x shape: (batch_size, channels, freq_bins, time_frames)
        for layer in self.layers:
            x = layer(x)
        return x


class InceptionLayer(nn.Module):
    """
    单个Inception层，包含多个不同大小的卷积核
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7]):
        super(InceptionLayer, self).__init__()
        
        self.branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            # 计算padding以保持特征图大小一致
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # 平均池化下采样
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # 并行处理所有分支
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 在通道维度上拼接
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # 平均池化下采样
        output = self.avgpool(concatenated)
        
        return output


class FFN(nn.Module):
    """
    前馈神经网络，用于最终输出
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(FFN, self).__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.ffn(x)


@register_model('SFFTDualBranchNet')
class SFFTDualBranchNet(nn.Module):
    """
    SFFT双分支网络模型
    包含SFFT特征提取、全卷积分支、Inception分支和FFN输出
    """
    def __init__(self, configs):
        super(SFFTDualBranchNet, self).__init__()
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        
        # SFFT特征提取
        self.sfft_extractor = SFFTFeatureExtractor(self.feature_size)
        
        # 计算SFFT输出的通道数（这里假设为1，因为是幅度谱）
        sfft_channels = 1
        
        # 全卷积分支
        self.full_conv_branch = FullConvBranch(
            in_channels=sfft_channels, 
            base_channels=64
        )
        
        # Inception分支
        self.inception_branch = InceptionBranch(
            in_channels=sfft_channels, 
            base_channels=32
        )
        
        # 计算展平后的特征维度（需要根据实际的SFFT输出大小来计算）
        # 这里使用一个示例输入来计算维度
        with torch.no_grad():
            dummy_input = torch.randn(1, self.feature_size)
            sfft_output = self.sfft_extractor(dummy_input)
            # 添加通道维度
            sfft_output = sfft_output.unsqueeze(1)  # (1, 1, freq_bins, time_frames)
            
            conv_output = self.full_conv_branch(sfft_output)
            inception_output = self.inception_branch(sfft_output)
            
            # 展平
            conv_flattened = conv_output.view(conv_output.size(0), -1)
            inception_flattened = inception_output.view(inception_output.size(0), -1)
            
            # 计算总的特征维度
            total_features = conv_flattened.size(1) + inception_flattened.size(1)
        
        # FFN输出层
        self.ffn = FFN(
            input_dim=total_features,
            hidden_dim=512,
            output_dim=self.label_size,
            dropout=0.1
        )
    
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
        # SFFT特征提取
        sfft_features = self.sfft_extractor(x_enc)  # (batch_size, freq_bins, time_frames)
        
        # 添加通道维度用于2D卷积
        sfft_features = sfft_features.unsqueeze(1)  # (batch_size, 1, freq_bins, time_frames)
        
        # 全卷积分支
        conv_features = self.full_conv_branch(sfft_features)
        
        # Inception分支
        inception_features = self.inception_branch(sfft_features)
        
        # 展平特征
        conv_flattened = conv_features.view(conv_features.size(0), -1)
        inception_flattened = inception_features.view(inception_features.size(0), -1)
        
        # 拼接两个分支的特征
        combined_features = torch.cat([conv_flattened, inception_flattened], dim=1)
        
        # FFN输出
        output = self.ffn(combined_features)
        
        return output