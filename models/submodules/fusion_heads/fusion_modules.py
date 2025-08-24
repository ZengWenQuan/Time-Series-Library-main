import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_fusion

class FeatureAdjuster(nn.Module):
    """
    A module to adjust the number of channels and sequence length of a feature tensor.
    """
    def __init__(self, in_channels, config):
        super(FeatureAdjuster, self).__init__()
        
        # Adjust channels if out_channels is specified
        out_channels = config.get('out_channels', in_channels)
        self.channel_adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.output_channels = out_channels

        # Adjust length if out_len is specified
        out_len = config.get('out_len')
        self.length_adjust = nn.AdaptiveAvgPool1d(out_len) if out_len is not None else nn.Identity()

    def forward(self, x):
        x = self.channel_adjust(x)
        x = self.length_adjust(x)
        return x

@register_fusion
class ConcatFusion(nn.Module):
    """简单的拼接融合，可处理一个向量和一个序列的融合"""
    def forward(self, seq_features, vec_features):
        # seq_features: [B, D_seq, L], vec_features: [B, D_vec]
        vec_expanded = vec_features.unsqueeze(1).expand(-1, seq_features.size(2), -1)
        seq_transposed = seq_features.transpose(1, 2) # -> [B, L, D_seq]
        fused = torch.cat([seq_transposed, vec_expanded], dim=-1)
        return fused

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

        # Create feature adjusters for each branch
        self.adjuster_norm = FeatureAdjuster(config['dim_norm'], config.get('adjustment_config_norm', {}))
        self.adjuster_cont = FeatureAdjuster(config['dim_cont'], config.get('adjustment_config_cont', {}))

        # The number of channels after adjustment
        adjusted_channels_norm = self.adjuster_norm.output_channels
        adjusted_channels_cont = self.adjuster_cont.output_channels

        # For 'add' and 'cross-attention', channels must match after adjustment
        if self.strategy in ['add', 'cross-attention'] and adjusted_channels_norm != adjusted_channels_cont:
            raise ValueError(f"融合策略 '{self.strategy}' 要求调整后的通道数一致, "
                             f"但得到: norm={adjusted_channels_norm}, cont={adjusted_channels_cont}")
        
        final_channels = adjusted_channels_norm

        # Initialize layers based on strategy
        if self.strategy == 'cross-attention':
            num_heads = config.get('attention_heads', 4)
            self.attention = nn.MultiheadAttention(embed_dim=final_channels, num_heads=num_heads, batch_first=True)
            self.output_dim = final_channels
        
        elif self.strategy == 'concat':
            concatenated_channels = adjusted_channels_norm + adjusted_channels_cont
            fusion_out_channels = config.get('out_channels', concatenated_channels)
            self.fusion_conv = nn.Conv1d(in_channels=concatenated_channels, out_channels=fusion_out_channels, kernel_size=1)
            self.output_dim = fusion_out_channels
        
        elif self.strategy != 'add':
            raise ValueError(f"未知的融合策略: '{self.strategy}'")
        else: # 'add' strategy
            self.output_dim = final_channels


    def forward(self, features_norm, features_cont):
        # 1. Adjust features from each branch
        features_norm = self.adjuster_norm(features_norm)
        features_cont = self.adjuster_cont(features_cont)

        # For 'add', shapes must be identical after adjustment
        if self.strategy == 'add' and features_norm.shape != features_cont.shape:
            raise ValueError(
                f"融合策略 'add' 要求调整后的两个分支输出形状完全一致。"
                f"但收到的形状不匹配: \n"
                f"  - 调整后归一化谱分支输出: {features_norm.shape}\
"
                f"  - 调整后连续谱分支输出: {features_cont.shape}")

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