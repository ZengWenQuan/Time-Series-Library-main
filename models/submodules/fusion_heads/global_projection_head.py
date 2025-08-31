import torch
import torch.nn as nn
from ...registries import register_head

@register_head
class GlobalProjectionHead(nn.Module):
    """
    全局特征投射头。

    设计思想:
    借鉴自 PatchTST, DLinear 等SOTA时间序列模型。其核心思想是，上游模块提取的
    特征图已经足够丰富，无需再进行复杂的序列处理。本模块将特征图在长度维度上
    进行全局平均池化，将所有序列信息聚合为一个特征向量，然后通过一个简单的FFN
    网络直接投射到最终的预测值。

    优点:
    - 简洁高效，参数少，计算快。
    - 全局池化对输入序列的平移和微小形变不敏感，更鲁棒。
    - 不易过拟合，是一个非常可靠的基线和通用预测头。
    """
    def __init__(self, config):
        super().__init__()
        self.targets = config['targets']
        dropout_rate = config.get('dropout_rate', 0.2)
        ffn_layers_config = config.get('ffn_layers', [256, 128])
        num_targets = len(self.targets)

        # --- 1. 全局平均池化层 ---
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # --- 2. FFN网络 ---
        # 使用 LazyLinear，无需在初始化时知道输入通道数，它会自动推断
        layers = [nn.LazyLinear(ffn_layers_config[0])]
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout_rate))
        
        current_dim = ffn_layers_config[0]
        for hidden_dim in ffn_layers_config[1:]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_targets))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        # x 的输入形状: (B, C, L)
        
        # 1. 全局池化
        x = self.global_pool(x) # -> (B, C, 1)
        
        # 2. 展平
        x = x.squeeze(-1) # -> (B, C)
        
        # 3. 通过FFN得到最终预测
        prediction = self.ffn(x)
        
        return prediction
