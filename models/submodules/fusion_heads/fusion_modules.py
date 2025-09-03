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
        self.output_channels = target_channels
        self.output_length = target_length

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
        self.output_channels = target_channels
        self.output_length = target_length

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
        self.output_channels = target_channels # 遵循风格
        self.output_length = target_length

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)
        
        query = features_norm.permute(0, 2, 1)
        key = features_cont.permute(0, 2, 1)
        value = features_cont.permute(0, 2, 1)
        
        attended_output, _ = self.attention(query, key, value)
        return attended_output.permute(0, 2, 1)

# --- 新增的融合模块 ---

@register_fusion
class FilmFusion(nn.Module):
    """
    融合模块：门控融合 (FiLM-style)。
    让一个分支（控制器）的特征去动态调制另一个分支（被调制者）的特征。
    """
    def __init__(self, config):
        super().__init__()
        if 'target_shape' not in config:
            raise ValueError("Fusion config must contain 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']
        self.controller_branch = config.get('controller_branch', 'continuum')

        self.adjuster_norm = FeatureAdjuster(target_channels, target_length)
        self.adjuster_cont = FeatureAdjuster(target_channels, target_length)

        ffn_dims = config.get('ffn_dims', [target_channels, target_channels * 2])
        controller_input_dim = target_channels * target_length
        
        layers = []
        current_dim = controller_input_dim
        for hidden_dim in ffn_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, target_channels * 2)) # *2 for gamma and beta
        self.controller_mlp = nn.Sequential(*layers)

        self.output_channels = target_channels
        self.output_length = target_length
        self.output_dim = target_channels

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)

        if self.controller_branch == 'continuum':
            controller_features = features_cont
            modulated_features = features_norm
        else:
            controller_features = features_norm
            modulated_features = features_cont

        B, C, L = controller_features.shape
        controller_flat = controller_features.view(B, -1)
        
        params = self.controller_mlp(controller_flat)
        gamma, beta = torch.chunk(params, 2, dim=-1)
        
        # Reshape gamma and beta for broadcasting
        gamma = gamma.view(B, C, 1)
        beta = beta.view(B, C, 1)

        return gamma * modulated_features + beta

@register_fusion
class GruFusion(nn.Module):
    """
    融合模块：循环融合 (GRU-based)。
    将两个分支的特征拼接后，通过一个双向GRU来捕捉序列依赖性并融合。
    """
    def __init__(self, config):
        super().__init__()
        if 'target_shape' not in config:
            raise ValueError("Fusion config must contain 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']

        self.adjuster_norm = FeatureAdjuster(target_channels, target_length)
        self.adjuster_cont = FeatureAdjuster(target_channels, target_length)

        gru_input_size = target_channels * 2
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 1)
        dropout = config.get('dropout_rate', 0.1)

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # The output of BiGRU is 2*hidden_size, we use a conv to map it back
        self.output_conv = nn.Conv1d(hidden_size * 2, target_channels, kernel_size=1)

        self.output_channels = target_channels
        self.output_length = target_length
        self.output_dim = target_channels

    def forward(self, features_norm, features_cont):
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)

        x = torch.cat([features_norm, features_cont], dim=1) # -> [B, 2*C, L]
        x = x.permute(0, 2, 1) # -> [B, L, 2*C]

        gru_out, _ = self.gru(x)
        
        # Permute back and map to target channels
        # gru_out: [B, L, H*2] -> [B, H*2, L]
        fused = gru_out.permute(0, 2, 1)
        fused = self.output_conv(fused)

        return fused