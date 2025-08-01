import torch
import torch.nn as nn
import torch.nn.functional as F


class SFFTFeatureExtractor(nn.Module):
    """
    短时傅里叶变换特征提取器
    """
    def __init__(self, args):
        super(SFFTFeatureExtractor, self).__init__()
        self.feature_size = args.feature_size
        self.n_fft = args.n_fft if hasattr(args, 'n_fft') else 128
        self.hop_length = args.hop_length if hasattr(args, 'hop_length') else 32
        # 计算SFFT后的特征维度
        self.time_bins = (self.feature_size - self.n_fft) // self.hop_length + 1
        self.freq_bins = self.n_fft // 2 + 1
        
    def forward(self, x):
        # 输入x的形状为 [batch_size, feature_size]
        complex_stft = torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(x.device),
            center=False,
            return_complex=True
        )
        
        magnitude = torch.abs(complex_stft)
        magnitude = torch.log1p(magnitude)
        return magnitude.unsqueeze(1)  # [batch_size, 1, freq_bins, time_bins]


class ConvBranch(nn.Module):
    """
    卷积分支，通过卷积网络处理时频特征
    """
    def __init__(self, args):
        super(ConvBranch, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        # 配置卷积通道数
        in_channels = 1
        conv_channels = [
            args.conv_channel_1 if hasattr(args, 'conv_channel_1') else 8,
            args.conv_channel_2 if hasattr(args, 'conv_channel_2') else 16,
            args.conv_channel_3 if hasattr(args, 'conv_channel_3') else 32
        ]
        pool_size = args.pool_size if hasattr(args, 'pool_size') else 2
        
        for i, channels in enumerate(conv_channels):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=pool_size, stride=2, padding=1)
            ))
            in_channels = channels
            
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception模块，用于多尺度特征提取
    """
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )
        
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(),
            nn.Conv2d(out_channels//4, out_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU()
        )
        
    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch3x3(x)
        branch3 = self.branch5x5(x)
        branch4 = self.branch7x7(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionBranch(nn.Module):
    """
    Inception分支，通过Inception模块处理时频特征
    """
    def __init__(self, args):
        super(InceptionBranch, self).__init__()
        self.inception_layers = nn.ModuleList()
        
        # 配置Inception通道数
        in_channels = 1
        inception_channels = [
            args.inception_channel_1 if hasattr(args, 'inception_channel_1') else 8,
            args.inception_channel_2 if hasattr(args, 'inception_channel_2') else 16,
            args.inception_channel_3 if hasattr(args, 'inception_channel_3') else 32
        ]
        pool_size = args.pool_size if hasattr(args, 'pool_size') else 2
        
        for i, channels in enumerate(inception_channels):
            self.inception_layers.append(nn.Sequential(
                InceptionModule(in_channels, channels),
                nn.AvgPool2d(kernel_size=pool_size, stride=2, padding=1)
            ))
            in_channels = channels
            
    def forward(self, x):
        for layer in self.inception_layers:
            x = layer(x)
        return x


class Model(nn.Module):
    """
    混合卷积和Inception模型，用于恒星参数估计
    """
    def __init__(self, args):
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.feature_size = args.feature_size
        self.label_size = args.label_size
        
        # SFFT特征提取器
        self.sfft = SFFTFeatureExtractor(args)
        
        # 计算SFFT后的特征维度
        n_fft = args.n_fft if hasattr(args, 'n_fft') else 128
        hop_length = args.hop_length if hasattr(args, 'hop_length') else 32
        time_bins = (self.feature_size - n_fft) // hop_length + 1
        freq_bins = n_fft // 2 + 1
        
        # 双分支结构
        self.conv_branch = ConvBranch(args)
        self.inception_branch = InceptionBranch(args)
        
        # 计算展平后的特征大小
        self.conv_output_size = self._calculate_output_size((1, freq_bins, time_bins), self.conv_branch)
        self.inception_output_size = self._calculate_output_size((1, freq_bins, time_bins), self.inception_branch)
        
        # 全连接网络
        total_features = self.conv_output_size + self.inception_output_size
        ffn_hidden_size = args.ffn_hidden_size if hasattr(args, 'ffn_hidden_size') else 64
        dropout_rate = args.dropout_rate if hasattr(args, 'dropout_rate') else 0.3
        
        self.ffn = nn.Sequential(
            nn.Linear(total_features, ffn_hidden_size),
            nn.BatchNorm1d(ffn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_hidden_size, self.label_size)
        )
        
    def _calculate_output_size(self, input_size, module):
        """计算网络输出特征的大小"""
        x = torch.rand(1, *input_size)
        x = module(x)
        return x.view(1, -1).size(1)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """符合项目架构的前向传播函数"""
        if self.task_name == 'regression' or self.task_name == 'stellar_parameter_estimation':
            return self.regression(x_enc)
        return None
    
    def regression(self, x):
        """
        执行回归任务
        x: [batch_size, seq_len]
        返回: [batch_size, pred_len]
        """
        # 提取SFFT特征
        x = self.sfft(x)
        
        # 双分支处理
        conv_out = self.conv_branch(x)
        inception_out = self.inception_branch(x)
        
        # 展平特征
        conv_flat = conv_out.view(conv_out.size(0), -1)
        inception_flat = inception_out.view(inception_out.size(0), -1)
        
        # 特征融合
        combined = torch.cat([conv_flat, inception_flat], dim=1)
        
        # 全连接层预测
        output = self.ffn(combined)
        
        return output