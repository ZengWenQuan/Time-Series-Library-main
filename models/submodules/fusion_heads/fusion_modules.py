import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_fusion

class FeatureAdjuster(nn.Module):
    """
    一个根据目标通道和长度调整特征张量的模块。
    """
    def __init__(self, in_channels, out_channels, out_len):
        super(FeatureAdjuster, self).__init__()
        
        # Adjust channels if necessary
        self.channel_adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Adjust length if necessary
        self.length_adjust = nn.AdaptiveAvgPool1d(out_len) if out_len is not None else nn.Identity()

    def forward(self, x):
        x = self.channel_adjust(x)
        x = self.length_adjust(x)
        return x


@register_fusion
class FusionModule(nn.Module):
    """
    通用融合模块 (v4)。
    在融合前，使用可配置的 FeatureAdjuster 调整每个输入分支的通道数和长度。
    支持 'add', 'concat', 'cross-attention' 策略。
    """
    def __init__(self, config):
        super(FusionModule, self).__init__()
        self.strategy = config.get('strategy', 'concat').lower()

        # Get target shape from config
        if 'target_shape' not in config:
            raise ValueError("FusionModule 的配置中必须包含 'target_shape'")
        
        target_channels = config['target_shape']['channels']
        target_length = config['target_shape']['length']

        # Create feature adjusters for each branch
        self.adjuster_norm = FeatureAdjuster(
            in_channels=config['channels_norm'],
            out_channels=target_channels,
            out_len=target_length
        )
        self.adjuster_cont = FeatureAdjuster(
            in_channels=config['channels_cont'],
            out_channels=target_channels,
            out_len=target_length
        )

        # Initialize layers based on strategy
        if self.strategy == 'cross-attention':
            num_heads = config.get('attention_heads', 4)
            self.attention = nn.MultiheadAttention(embed_dim=target_channels, num_heads=num_heads, batch_first=True)
            self.output_dim = target_channels
        
        elif self.strategy == 'concat':
            # After adjustment, both branches have `target_channels`
            concatenated_channels = target_channels * 2
            fusion_out_channels = config.get('out_channels', concatenated_channels)
            self.fusion_conv = nn.Conv1d(in_channels=concatenated_channels, out_channels=fusion_out_channels, kernel_size=1)
            self.output_dim = fusion_out_channels
        
        elif self.strategy == 'add':
            self.output_dim = target_channels
        
        else:
            raise ValueError(f"未知的融合策略: '{self.strategy}'")


    def forward(self, features_norm, features_cont):
        # 1. Adjust features from each branch to the same target shape
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)

        # 2. Fuse based on strategy
        if self.strategy == 'add':
            return features_norm + features_cont
        
        elif self.strategy == 'concat':
            return self.fusion_conv(torch.cat([features_norm, features_cont], dim=1))

        elif self.strategy == 'cross-attention':
            # Input shape for MultiheadAttention: (B, L, C)
            query = features_norm.permute(0, 2, 1)
            key = features_cont.permute(0, 2, 1)
            value = features_cont.permute(0, 2, 1)
            
            attended_output, _ = self.attention(query, key, value)
            
            # Revert to (B, C, L)
            return attended_output.permute(0, 2, 1)



@register_fusion
class CrossAttentionFusion(nn.Module):
    def __init__(self, config):
        super(CrossAttentionFusion, self).__init__()
        continuum_dim, absorption_dim, fusion_dim = config['continuum_dim'], config['absorption_dim'], config['fusion_dim']
        num_heads, dropout_rate = config.get('num_heads', 4), config.get('dropout', 0.1)
        self.continuum_proj = nn.Linear(continuum_dim, fusion_dim)
        self.absorption_proj = nn.Linear(absorption_dim, fusion_dim)
        self.norm_cont = nn.LayerNorm(fusion_dim)
        self.norm_abs = nn.LayerNorm(fusion_dim)
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads, batch_first=True, dropout=dropout_rate)
        self.fusion_layer = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
        
    def forward(self, continuum_features, absorption_features):
        cont_proj = self.continuum_proj(continuum_features).unsqueeze(1)
        abs_proj = self.absorption_proj(absorption_features).unsqueeze(1)
        norm_cont_proj, norm_abs_proj = self.norm_cont(cont_proj), self.norm_abs(abs_proj)
        cont_attended, _ = self.cross_attention(norm_cont_proj, norm_abs_proj, norm_abs_proj)
        abs_attended, _ = self.cross_attention(norm_abs_proj, norm_cont_proj, norm_cont_proj)
        return self.fusion_layer(torch.cat([cont_attended.squeeze(1), abs_attended.squeeze(1)], dim=1))