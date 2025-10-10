import torch
import torch.nn as nn
from ..registries import register_head

@register_head
class SimpleFFNMultiTaskHead(nn.Module):
    """
    一个简单的多任务预测头。
    
    - 对3D输入 (B, C, L) 使用自适应池化，然后通过独立的FFN预测每个任务。
    - 对2D输入 (B, F) 直接通过FFN预测。
    - 每个FFN包含一个隐藏层。
    """
    def __init__(self, config):
        super().__init__()
        
        self.targets = config['targets']
        self.pool = None
        
        # 根据输入类型确定FFN的输入维度
        if 'input_channels' in config:
            # 假设是3D输入，将使用自适应池化
            in_features = config['input_channels']
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif 'in_features' in config:
            # 假设是2D输入
            in_features = config['in_features']
        else:
            raise ValueError("Head config must contain 'input_channels' (for 3D input) or 'in_features' (for 2D input)")

        hidden_dim = config.get('hidden_dim', 128) # 单个隐藏层的维度
        dropout_rate = config.get('dropout', 0.1)

        self.heads = nn.ModuleDict()

        for target_name in self.targets:
            self.heads[target_name] = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        # 如果是3D输入且定义了池化层，则先池化
        x = self.pool(x) # (B, C, 1)
        
        # 展平特征以送入FFN
        x = torch.flatten(x, 1)

        # 通过各自的头进行预测
        outputs = []
        for target_name in self.targets:
            prediction = self.heads[target_name](x)
            outputs.append(prediction)
        
        # 拼接结果
        return torch.cat(outputs, dim=1)
