import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from exp.exp_basic import register_model

class DWTFeature(nn.Module):
    """
    小波DWT特征提取模块，将输入序列分解为多尺度特征。
    """
    def __init__(self, wavelet='db1', level=3):
        super().__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        # x: (batch, seq_len)
        coeffs = []
        for i in range(x.shape[0]):
            c = pywt.wavedec(x[i].cpu().numpy(), self.wavelet, level=self.level)
            c = [torch.tensor(ci, dtype=x.dtype, device=x.device) for ci in c]
            coeffs.append(torch.cat([ci.flatten() for ci in c], dim=0))
        coeffs = torch.stack(coeffs, dim=0)
        return coeffs  # (batch, dwt_feat_len)

class WaveletConvBlock(nn.Module):
    """
    多尺度卷积+池化块
    """
    def __init__(self, in_ch, out_ch, kernel_sizes=[3,5,7], pool_size=2):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, (k, k), padding=(k//2, k//2)) for k in kernel_sizes
        ])
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        # x: (batch, in_ch, H, W)
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)  # 通道拼接
        out = self.pool(out)
        return out

@register_model('WaveletConvNet')
class WaveletConvNet(nn.Module):
    """
    基于小波和多尺度卷积的恒星参数估计模型
    输入: (batch, seq_len)
    输出: (batch, 4)
    """
    def __init__(self, configs):
        super(WaveletConvNet, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dwt_level = configs.dwt_level
        self.wavelet = configs.wavelet
        self.ffn_dim = configs.ffn_dim
        self.dwt = DWTFeature(wavelet=self.wavelet, level=self.dwt_level)
        # 计算DWT特征长度
        dummy = torch.zeros(self.seq_len)
        coeffs = pywt.wavedec(dummy.numpy(), self.wavelet, level=self.dwt_level)
        dwt_feat_len = sum([len(c) for c in coeffs])
        # 取最近的平方数reshape成方阵
        side = int(dwt_feat_len ** 0.5)
        self.feat_map_size = side
        self.dwt_feat_len = dwt_feat_len
        self.reshape_len = side * side
        self.in_ch = 1
        self.out_ch = 8
        # 三层多尺度卷积
        self.conv1 = WaveletConvBlock(self.in_ch, self.out_ch)
        self.conv2 = WaveletConvBlock(self.out_ch*3, self.out_ch)
        self.conv3 = WaveletConvBlock(self.out_ch*3, self.out_ch)
        # FFN
        self.ffn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.out_ch*3 * (side//8)**2, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, 4)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (batch, seq_len)
        feat = self.dwt(x_enc)  # (batch, dwt_feat_len)
        # 截断或补零reshape成(batch, 1, H, W)
        if feat.shape[1] < self.reshape_len:
            pad = feat.new_zeros((feat.shape[0], self.reshape_len - feat.shape[1]))
            feat = torch.cat([feat, pad], dim=1)
        else:
            feat = feat[:, :self.reshape_len]
        feat = feat.view(-1, 1, self.feat_map_size, self.feat_map_size)
        out = self.conv1(feat)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.ffn(out)
        return out

if __name__ == "__main__":
    # 简单测试
    model = Model(configs)
    x = torch.randn(2, 4802)
    y = model(x)
    print(y.shape)  # 应为(batch, 4) 