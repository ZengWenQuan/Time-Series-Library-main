import torch
import torch.nn as nn
from ..registries import register_head

@register_head
class SimpleFFNMultiTaskHead(nn.Module):
    """
    一个简单的多任务预测头。
    
    该模块接收融合后的特征，并为每个目标参数使用一个独立的
    前馈网络 (FFN) 来进行预测。
    输入特征应为2D张量 (batch_size, in_features)。
    """
    def __init__(self, config):
        super().__init__()
        
        # 从主配置中获取目标列表和头部特定配置
        self.targets = config['targets']
        head_config = config['head_config']

        # 从头部配置中获取参数
        in_features = head_config['in_features']
        hidden_dims = head_config.get('hidden_dims', [256, 128]) # 提供默认值
        dropout_rate = head_config.get('dropout', 0.1)

        self.heads = nn.ModuleDict()

        # 为每个目标参数创建一个独立的FFN
        for target_name in self.targets:
            layers = []
            current_dim = in_features

            # 构建隐藏层
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
                current_dim = h_dim
            
            # 添加最终的输出层，输出维度为1
            layers.append(nn.Linear(current_dim, 1))
            
            self.heads[target_name] = nn.Sequential(*layers)

    def forward(self, x):
        # 确保输入是2D的 (B, Features)
        if x.dim() > 2:
            x = torch.flatten(x, 1)

        # 分别通过各自的FFN头进行预测
        outputs = []
        for target_name in self.targets:
            prediction = self.heads[target_name](x)
            outputs.append(prediction)
        
        # 将所有预测结果拼接成 (B, num_targets) 的张量
        return torch.cat(outputs, dim=1)
