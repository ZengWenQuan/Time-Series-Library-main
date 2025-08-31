import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from ...registries import register_normalized_branch

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesBlock(nn.Module):
    def __init__(self, seq_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = 0 # Not used for feature extraction
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

@register_normalized_branch
class TimesNetBranch(nn.Module):
    """
    基于 TimesNet 的特征提取分支 (v2, 遵循项目风格)。
    """
    def __init__(self, config):
        super().__init__()
        # --- 1. 获取配置参数 ---
        input_len = config['input_len']
        d_model = config.get('d_model', 64)
        e_layers = config.get('e_layers', 2)
        top_k = config.get('top_k', 5)
        d_ff = config.get('d_ff', 64)
        num_kernels = config.get('num_kernels', 6)
        dropout = config.get('dropout_rate', 0.1)

        # --- 2. 初始化模块 ---
        # 2.1 Embedding layer to project input to d_model
        # The input is (B, 1, L), we want to get (B, d_model, L)
        self.embedding = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=1)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 2.2 TimesNet backbone
        self.model = nn.ModuleList([
            TimesBlock(
                seq_len=input_len,
                top_k=top_k,
                d_model=d_model,
                d_ff=d_ff,
                num_kernels=num_kernels
            ) for _ in range(e_layers)
        ])
        
        # --- 3. 保存输出维度信息 ---
        self.output_channels = d_model
        self.output_length = input_len # TimesNet preserves length

    def forward(self, x):
        # x: [B, C, L], C=1 for spectra
        if x.shape[1] != 1:
            raise ValueError("TimesNetBranch expects input with 1 channel.")

        # Normalization
        means = x.mean(2, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # Embedding: [B, 1, L] -> [B, d_model, L]
        enc_out = self.dropout(self.embedding(x))
        
        # Permute for TimesBlock: [B, d_model, L] -> [B, L, d_model]
        enc_out = enc_out.permute(0, 2, 1)

        # TimesNet backbone
        for i in range(len(self.model)):
            enc_out = self.layer_norm_in(self.model[i](enc_out))
        
        # Permute back to [B, C, L] format
        return enc_out.permute(0, 2, 1)