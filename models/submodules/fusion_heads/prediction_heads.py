
import torch
import torch.nn as nn
import torch.nn.init as init
from ...registries import register_head

@register_head
class LSTMHead(nn.Module):
    """通用的 LSTM + FFN 预测头"""
    def __init__(self, config):
        super(LSTMHead, self).__init__()
        self.targets = config['targets']
        dropout_rate = config.get('dropout_rate', 0.2)
        self.bilstm = nn.LSTM(config['head_input_dim'], config['lstm_hidden_dim'], config['lstm_layers'], batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        head_input_dim = config['lstm_hidden_dim'] * 2
        self.prediction_heads = nn.ModuleDict()
        for target in self.targets:
            layers, in_features = [], head_input_dim
            for out_features in config['prediction_head']['hidden_layers']:
                layers.extend([nn.Linear(in_features, out_features), nn.ReLU()])
                in_features = out_features
            layers.append(nn.Linear(in_features, 1))
            self.prediction_heads[target] = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        # 新增：将输入的 (B, C, L) 格式转为LSTM需要的 (B, L, C) 格式
        x = x.transpose(1, 2)

        # x可以是融合后的序列 [B, L, D_fused] 或 单分支序列 [B, L, D_branch]
        lstm_out, _ = self.bilstm(x)
        final_features = lstm_out[:, -1, :]
        predictions = [self.prediction_heads[target](final_features) for target in self.targets]
        return torch.cat(predictions, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.LSTM): 
                for param in m.parameters():
                    if len(param.shape) >= 2: init.orthogonal_(param.data)
                    else: init.normal_(param.data)

@register_head
class FFNHead(nn.Module):
    """简单的FFN预测头，用于已经提取了全局特征的场景"""
    def __init__(self, config):
        super(FFNHead, self).__init__()
        self.targets = config['targets']
        self.label_size = len(self.targets)
        self.use_separate_heads = config.get('use_separate_heads', False)

        if self.use_separate_heads:
            self.prediction_heads = nn.ModuleList()
            for _ in range(self.label_size):
                self.prediction_heads.append(self._create_ffn_block(config, 1))
        else:
            self.prediction_head = self._create_ffn_block(config, self.label_size)
        
        self._initialize_weights()

    def _create_ffn_block(self, config, output_dim):
        fc_layers = []
        current_dim = config['ffn_input_dim']
        use_batch_norm = config.get('use_batch_norm', True)
        dropout_rate = config.get('dropout_rate', 0.2)

        for hidden_dim in config['fc_hidden_dims']:
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        fc_layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*fc_layers)

    def forward(self, x):
        if self.use_separate_heads:
            predictions = [head(x) for head in self.prediction_heads]
            return torch.cat(predictions, dim=1)
        else:
            return self.prediction_head(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

@register_head
class AttentionExpertHead(nn.Module):
    """
    结合注意力池化和独立专家网络的预测头。
    1. 使用注意力机制将序列特征池化为单个向量。
    2. 使用独立的MLP（专家）为每个目标进行预测。
    """
    def __init__(self, config):
        super(AttentionExpertHead, self).__init__()
        self.targets = config['targets']
        head_input_dim = config['head_input_dim']
        dropout_rate = config.get('dropout_rate', 0.2)
        
        # 注意力池化层
        self.attention = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.Tanh(),
            nn.Linear(head_input_dim // 2, 1)
        )
        
        # 独立专家网络
        self.expert_heads = nn.ModuleDict()
        for target in self.targets:
            layers = []
            in_features = head_input_dim
            # 从配置中读取专家网络的隐藏层维度
            for out_features in config.get('expert_hidden_layers', [in_features // 2, in_features // 4]):
                layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                in_features = out_features
            layers.append(nn.Linear(in_features, 1))
            self.expert_heads[target] = nn.Sequential(*layers)
            
        self._initialize_weights()

    def forward(self, x):
        # 输入 x 的形状: (B, C, L)
        # 转置以匹配注意力计算: (B, L, C)
        x_transposed = x.transpose(1, 2)
        
        # 计算注意力权重
        # energy shape: (B, L, 1)
        energy = self.attention(x_transposed)
        # weights shape: (B, L, 1)
        weights = torch.softmax(energy, dim=1)
        
        # 应用注意力权重进行池化
        # context shape: (B, C)
        context = torch.sum(x_transposed * weights, dim=1)
        
        # 通过独立的专家网络进行预测
        predictions = [self.expert_heads[target](context) for target in self.targets]
        
        return torch.cat(predictions, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
