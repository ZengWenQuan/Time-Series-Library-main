import torch
import torch.nn as nn
from ...registries import register_head

@register_head
class MultiTaskHead(nn.Module):
    """
    多任务预测头 (v5)。
    使用懒加载(LazyLinear)来自动推断维度，实现卷积主干和FFN头的无缝衔接。
    """
    def __init__(self, config, targets):
        super(MultiTaskHead, self).__init__()
        self.targets = targets
        
        # --- 1. 构建共享的卷积下采样主干 ---
        conv_config = config['conv_pyramid']
        conv_layers = []
        in_channels = conv_config['in_channels']
        
        for layer_conf in conv_config['layers']:
            out_channels = layer_conf['out_channels']
            conv_layers.append(nn.Conv1d(
                in_channels, out_channels, 
                kernel_size=layer_conf['kernel_size'], 
                padding=(layer_conf['kernel_size'] - 1) // 2
            ))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = out_channels
        self.shared_backbone = nn.Sequential(*conv_layers)

        # --- 2. 构建使用LazyLinear的独立多任务FFN头 ---
        self.heads = nn.ModuleDict()
        ffn_layers_config = config.get('ffn_layers', [128, 64])
        dropout_rate = config.get('dropout', 0.2)

        for target_name in self.targets:
            layers = []
            # 第一个线性层使用LazyLinear，无需提供in_features
            layers.append(nn.LazyLinear(out_features=ffn_layers_config[0]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # 后续的线性层
            current_dim = ffn_layers_config[0]
            for hidden_dim in ffn_layers_config[1:]:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                current_dim = hidden_dim
            
            # 最终的输出层
            layers.append(nn.Linear(current_dim, 1))
            self.heads[target_name] = nn.Sequential(*layers)

    def forward(self, x):
        # x 的输入形状是 (B, C, L)
        
        # 1. 通过共享卷积主干
        x = self.shared_backbone(x)
        
        # 2. 展平
        x = x.view(x.size(0), -1)
        
        # 3. 通过各自的FFN头
        # 第一次调用时，LazyLinear会根据x的维度自动确定其输入大小
        ordered_predictions = []
        for target_name in self.targets:
            head_model = self.heads[target_name]
            prediction = head_model(x)
            ordered_predictions.append(prediction)
        
        return torch.cat(ordered_predictions, dim=1)
