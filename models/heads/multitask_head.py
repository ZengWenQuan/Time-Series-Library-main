import torch
import torch.nn as nn
from ..registries import register_head
from ..submodules.fusion_heads.fusion_modules import FeatureAdjuster

@register_head
class MultiTaskHead(nn.Module):
    """
    多任务预测头 (v9)。
    从外部引入 FeatureAdjuster，在模块入口处自动调整输入特征的维度。
    """
    def __init__(self, config):
        super(MultiTaskHead, self).__init__()
        self.targets = config['targets']
        use_batch_norm = config.get('use_batch_norm')
        dropout_rate = config.get('dropout_rate')
        in_channels = config['input_channels']
        in_len = config['input_length']

        # --- 0. 入口特征调整器 ---
        self.adjuster = FeatureAdjuster(out_channels=in_channels, out_len=in_len)

        # --- 1. 自注意力模块 ---
        attention_heads = config.get('self_attention_heads', 4)
        self.attention_norm = nn.LayerNorm(in_channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=attention_heads, 
            batch_first=True
        )

        # --- 2. 构建共享的卷积下采样主干 --- 
        conv_config = config['conv_pyramid']
        conv_layers = []
        current_channels = in_channels
        current_len = in_len
        
        for layer_conf in conv_config['layers']:
            out_channels = layer_conf['out_channels']
            conv_layers.append(nn.Conv1d(
                current_channels, out_channels, 
                kernel_size=layer_conf['kernel_size'], 
                padding=(layer_conf['kernel_size'] - 1) // 2
            ))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            
            pool_size = layer_conf.get('pool_size', 1)
            conv_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            
            current_channels = out_channels
            current_len = current_len // pool_size

        self.shared_backbone = nn.Sequential(*conv_layers)

        # --- 3. 构建使用精确维度的独立多任务FFN头 ---
        ffn_input_dim = current_channels * current_len
        self.heads = nn.ModuleDict()
        ffn_layers_config = config.get('ffn_layers', [128, 64])

        for target_name in self.targets:
            layers = []
            layers.append(nn.Linear(ffn_input_dim, ffn_layers_config[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            current_dim = ffn_layers_config[0]
            for hidden_dim in ffn_layers_config[1:]:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                current_dim = hidden_dim
            
            layers.append(nn.Linear(current_dim, 1))
            self.heads[target_name] = nn.Sequential(*layers)

    def forward(self, x):
        # x 的输入形状是 (B, C_in, L_in)
        
        # 0. 自动调整输入维度
        x = self.adjuster(x) # 输出形状 (B, in_channels, in_len)
        
        # 1. 自注意力模块
        x_permuted = x.permute(0, 2, 1) # (B, in_len, in_channels)
        x_norm = self.attention_norm(x_permuted)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm)
        # 残差连接
        x = x_permuted + attn_output
        x = x.permute(0, 2, 1) # 转换回 (B, in_channels, in_len)
        
        # 2. 通过共享卷积主干
        x = self.shared_backbone(x)
        
        # 3. 展平
        x = x.view(x.size(0), -1)
        
        # 4. 通过各自的FFN头
        ordered_predictions = []
        for target_name in self.targets:
            head_model = self.heads[target_name]
            prediction = head_model(x)
            ordered_predictions.append(prediction)
        
        return torch.cat(ordered_predictions, dim=1)
