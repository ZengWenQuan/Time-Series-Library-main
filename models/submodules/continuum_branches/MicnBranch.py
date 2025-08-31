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
class MicnBranch(nn.Module):
    """
    基于 MICN 的特征提取分支。
    设计思想借鉴自 `models.other.MICN`。
    它使用多尺度卷积和等距卷积来提取特征。
    """
    def __init__(self, config):
        super().__init__()
        # --- 1. 获取配置参数 ---
        input_len = config['input_len']
        d_model = config.get('d_model', 64)
        d_ff = config.get('d_ff', 128)
        dropout = config.get('dropout_rate', 0.1)
        self.conv_kernel = config.get('conv_kernel', [12, 16])
        isometric_kernel = config.get('isometric_kernel', [18, 6])

        # --- 2. 初始化模块 ---
        # 2.1 Embedding layer
        self.embedding = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)

        # 2.2 MICN核心层
        # 等距卷积
        self.isometric_conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                                       kernel_size=i, padding=0, stride=1)
                                             for i in isometric_kernel])

        # 下采样卷积
        self.conv = nn.ModuleList([nn.Conv1d(in_channels=d_model, out_channels=d_model,
                                             kernel_size=i, padding=i // 2, stride=i)
                                   for i in self.conv_kernel])

        # 上采样卷积
        self.conv_trans = nn.ModuleList([nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in self.conv_kernel])

        # 分解模块
        self.decomp = nn.ModuleList([series_decomp(k) for k in [k+1 if k%2==0 else k for k in self.conv_kernel]])
        
        # 合并模块
        self.merge = torch.nn.Conv2d(in_channels=d_model, out_channels=d_model,
                                     kernel_size=(len(self.conv_kernel), 1))

        # 前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.norm = torch.nn.LayerNorm(d_model)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)

        # --- 3. 保存输出维度信息 ---
        self.output_channels = d_model
        self.output_length = input_len # MICN preserves length

    def _conv_trans_conv(self, x, conv1d, conv1d_trans, isometric):
        batch, seq_len, channel = x.shape
        permuted_x = x.permute(0, 2, 1)

        # Downsampling convolution
        x1 = self.drop(self.act(conv1d(permuted_x)))
        
        # Isometric convolution
        iso_kernel_size = isometric.kernel_size[0]
        pad_size = iso_kernel_size - 1
        zeros = torch.zeros((x1.shape[0], x1.shape[1], pad_size), device=x.device)
        iso_input = torch.cat((zeros, x1), dim=-1)
        iso_x = self.drop(self.act(isometric(iso_input)))
        
        # Residual connection and normalization
        x_iso = self.norm((iso_x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # Upsampling convolution
        x_up = self.drop(self.act(conv1d_trans(x_iso)))
        x_up = x_up[:, :, :seq_len]

        # Final residual connection and normalization
        output = self.norm(x_up.permute(0, 2, 1) + x)
        return output

    def forward(self, x_in):
        # x_in: [B, C, L], C=1 for spectra
        if x_in.shape[1] != 1:
            raise ValueError("MicnBranch expects input with 1 channel.")

        # 1. Embedding: [B, 1, L] -> [B, d_model, L]
        x = self.dropout_layer(self.embedding(x_in))

        # Permute for MICN: [B, d_model, L] -> [B, L, d_model]
        x = x.permute(0, 2, 1)

        # 2. MICN多尺度特征提取
        multi_scale_outputs = []
        for i in range(len(self.conv_kernel)):
            # Decompose into seasonal and trend components
            seasonal_comp, _ = self.decomp[i](x)
            
            # Apply MICN conv block
            processed_comp = self._conv_trans_conv(
                seasonal_comp, self.conv[i], self.conv_trans[i], self.isometric_conv[i]
            )
            multi_scale_outputs.append(processed_comp)

        # 3. 合并多尺度特征
        # Stack along a new dimension: [B, num_kernels, L, d_model]
        merged_output = torch.stack(multi_scale_outputs, dim=1)
        
        # Permute and merge: [B, d_model, num_kernels, L] -> [B, d_model, 1, L] -> [B, L, d_model]
        merged_output = self.merge(merged_output.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        # 4. FFN和残差连接
        y = self.norm1(merged_output)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)
        final_output = self.norm2(merged_output + y)

        # Permute back to [B, C, L] format
        return final_output.permute(0, 2, 1)