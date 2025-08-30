import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_fusion

class FeatureAdjuster(nn.Module):
    """
    一个根据目标通道和长度调整特征张量的模块。
    使用 LazyConv1d，因此只需要在初始化时定义输出通道。
    """
    def __init__(self, out_channels, out_len):
        super(FeatureAdjuster, self).__init__()
        # 使用 LazyConv1d，它会自动推断输入通道数
        self.channel_adjust = nn.LazyConv1d(out_channels, kernel_size=1)
        self.length_adjust = nn.AdaptiveAvgPool1d(out_len) if out_len is not None else nn.Identity()

    def forward(self, x):
        x = self.channel_adjust(x)
        x = self.length_adjust(x)
        return x

# --- 分解后的新模块 ---

@register_fusion
class add(nn.Module):
    """
    融合模块：逐元素相加。
    在融合前，使用 FeatureAdjuster 调整每个输入分支的通道数和长度。
    """
    def __init__(self, config):
        super().__init__()
        if 'target_shape' not in config:
            raise ValueError("Fusion config must contain 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']

        self.adjuster_norm = FeatureAdjuster(target_channels, target_length)
        self.adjuster_cont = FeatureAdjuster(target_channels, target_length)
        self.output_dim = target_channels

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)
        return features_norm + features_cont

@register_fusion
class concat(nn.Module):
    """
    融合模块：拼接融合。
    在融合前，使用 FeatureAdjuster 调整每个输入分支的通道数和长度。
    """
    def __init__(self, config):
        super().__init__()
        if 'target_shape' not in config:
            raise ValueError("Fusion config must contain 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']

        self.adjuster_norm = FeatureAdjuster(target_channels, target_length)
        self.adjuster_cont = FeatureAdjuster(target_channels, target_length)
        
        concatenated_channels = target_channels * 2
        self.fusion_conv = nn.Conv1d(in_channels=concatenated_channels, out_channels=target_channels, kernel_size=1)
        self.output_dim = target_channels

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)
        return self.fusion_conv(torch.cat([features_norm, features_cont], dim=1))

@register_fusion
class crossion_attention(nn.Module):
    """
    融合模块：交叉注意力机制。
    在融合前，使用 FeatureAdjuster 调整每个输入分支的通道数和长度。
    """
    def __init__(self, config):
        super().__init__()
        if 'target_shape' not in config:
            raise ValueError("Fusion config must contain 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']
        num_heads = config.get('attention_heads', 4)

        self.adjuster_norm = FeatureAdjuster(target_channels, target_length)
        self.adjuster_cont = FeatureAdjuster(target_channels, target_length)
        
        self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=num_heads, batch_first=True)
        self.output_dim = target_channels

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)
        
        query = features_norm.permute(0, 2, 1)
        key = features_cont.permute(0, 2, 1)
        value = features_cont.permute(0, 2, 1)
        
        attended_output, _ = self.attention(query, key, value)
        return attended_output.permute(0, 2, 1)

