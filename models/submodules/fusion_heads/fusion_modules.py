
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import register_fusion

@register_fusion('ConcatFusion')
class ConcatFusion(nn.Module):
    """简单的拼接融合，可处理一个向量和一个序列的融合"""
    def forward(self, seq_features, vec_features):
        # seq_features: [B, D_seq, L], vec_features: [B, D_vec]
        vec_expanded = vec_features.unsqueeze(1).expand(-1, seq_features.size(2), -1)
        seq_transposed = seq_features.transpose(1, 2) # -> [B, L, D_seq]
        fused = torch.cat([seq_transposed, vec_expanded], dim=-1)
        return fused

@register_fusion('GeneralFusion')
class FusionModule(nn.Module):
    """更通用的融合模块，支持add, concat, attention"""
    def __init__(self, config):
        super(FusionModule, self).__init__()
        self.strategy = config.get('strategy', 'concat').lower()
        dim_norm, dim_cont = config['dim_norm'], config['dim_cont']
        if self.strategy == 'add':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.output_dim = dim_norm
        elif self.strategy == 'attention':
            self.project_cont = nn.Linear(dim_cont, dim_norm)
            self.attention = nn.MultiheadAttention(dim_norm, config.get('attention_heads', 4), batch_first=True)
            self.output_dim = dim_norm
        else: self.output_dim = dim_norm + dim_cont

    def forward(self, features_norm, features_cont):
        features_norm = features_norm.transpose(1, 2)
        features_cont_expanded = features_cont.unsqueeze(1).expand(-1, features_norm.size(1), -1)
        if self.strategy == 'add': return features_norm + self.project_cont(features_cont_expanded)
        elif self.strategy == 'attention':
            projected_cont = self.project_cont(features_cont_expanded)
            return self.attention(features_norm, projected_cont, projected_cont, need_weights=False)[0]
        else: return torch.cat([features_norm, features_cont_expanded], dim=-1)

@register_fusion('CrossAttentionFusion')
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
