
import torch
import torch.nn as nn
import math
from models.registries import register_backbone

class PositionalEncoding(nn.Module):
    """为序列添加位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2) # 形状变为 (1, d_model, max_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的形状: (batch, channels, length)
        return x + self.pe[:, :, :x.size(2)]

@register_backbone
class HybridTransformerBackbone(nn.Module):
    """混合CNN-Transformer主干网络"""
    def __init__(self, cfg):
        super().__init__()
        in_channels = cfg['input_channels']
        input_length = cfg['input_length']
        stem_cfg = cfg['stem']
        transformer_cfg = cfg['transformer']
        use_batch_norm = cfg.get('use_batch_norm', True)

        current_length = input_length

        # --- CNN Stem --- 
        cnn_layers = []
        current_channels = in_channels
        for layer_cfg in stem_cfg['layers']:
            cnn_layers.append(nn.Conv1d(
                current_channels, layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=layer_cfg['padding']
            ))
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(layer_cfg['out_channels']))
            cnn_layers.append(nn.ReLU(True))
            
            current_channels = layer_cfg['out_channels']
            current_length = self._calculate_output_length(current_length, layer_cfg['kernel_size'], layer_cfg['stride'], layer_cfg['padding'])
        
        self.cnn_stem = nn.Sequential(*cnn_layers)

        # --- Transformer --- 
        d_model = transformer_cfg['d_model']
        # 如果CNN输出通道与d_model不匹配，则添加一个1x1卷积进行投影
        if current_channels != d_model:
            self.projection = nn.Conv1d(current_channels, d_model, 1)
        else:
            self.projection = nn.Identity()
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=transformer_cfg['nhead'],
            dim_feedforward=transformer_cfg['dim_feedforward'],
            dropout=transformer_cfg['dropout'],
            batch_first=True # 重要：我们的数据格式是 (Batch, Seq, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_cfg['num_layers'])

        self.output_channels = d_model
        self.output_length = current_length

    def _calculate_output_length(self, L_in, kernel_size, stride, padding, dilation=1):
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        # 1. CNN Stem
        x = self.cnn_stem(x) # (B, C_out, L_out)
        
        # 2. Projection to d_model
        x = self.projection(x) # (B, d_model, L_out)

        # 3. Add positional encoding
        x = self.pos_encoder(x) # (B, d_model, L_out)

        # 4. Transformer Encoder
        # Transformer需要 (Batch, Seq, Features)格式，所以需要转置
        x = x.permute(0, 2, 1) # (B, L_out, d_model)
        x = self.transformer_encoder(x)
        # 转置回来以匹配后续模块的期望输入 (Batch, Channels, Length)
        x = x.permute(0, 2, 1) # (B, d_model, L_out)
        
        return x
