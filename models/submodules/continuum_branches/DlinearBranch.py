import torch
import torch.nn as nn

from ...registries import register_continuum_branch


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
@register_continuum_branch
class DlinearBranch(nn.Module):
    """
    基于 DLinear 的特征提取分支。
    设计思想借鉴自 `models.other.DLinear`。
    它首先通过移动平均将序列分解为趋势项和季节项，然后分别用一个线性层处理，
    最后相加得到输出。非常适合捕捉光谱的平滑连续谱。
    """
    def __init__(self, config):
        super().__init__()
        # --- 1. 获取配置参数 ---
        input_len = config['input_len']
        # DLinear的输出长度是可配置的，而不是由模型结构固定
        output_len = config.get('output_len', 128) 
        individual = config.get('individual', False)
        moving_avg = config.get('moving_avg', 25)
        
        # DLinear处理的是单通道特征
        self.channels = 1 

        # --- 2. 初始化模块 ---
        self.decompsition = series_decomp(moving_avg)
        self.individual = individual

        if self.individual:
            # This part is kept for completeness but not used for single-channel spectra
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(input_len, output_len))
                self.Linear_Trend.append(nn.Linear(input_len, output_len))
        else:
            self.Linear_Seasonal = nn.Linear(input_len, output_len)
            self.Linear_Trend = nn.Linear(input_len, output_len)

        # --- 3. 保存输出维度信息 ---
        self.output_channels = self.channels
        self.output_length = output_len

    def forward(self, x):
        # x: [B, C, L], C=1 for spectra
        if x.shape[1] != 1:
            raise ValueError("DlinearBranch expects input with 1 channel.")

        seasonal_init, trend_init = self.decompsition(x)
        
        # Permute to put length last for Linear layer
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        if self.individual:
            seasonal_output = torch.zeros_like(seasonal_init)
            trend_output = torch.zeros_like(trend_init)
            seasonal_output[:, :, 0] = self.Linear_Seasonal[0](seasonal_init[:, :, 0])
            trend_output[:, :, 0] = self.Linear_Trend[0](trend_init[:, :, 0])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
        output = seasonal_output + trend_output
        
        # Permute back to [B, C, L] format
        return output.permute(0, 2, 1)
