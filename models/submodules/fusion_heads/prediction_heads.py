
import torch
import torch.nn as nn
import torch.nn.init as init
from ...registries import register_head

@register_head
class LSTMHead(nn.Module):
    """通用的 LSTM + FFN 预测头"""
    def __init__(self, config, targets):
        super(LSTMHead, self).__init__()
        self.targets = targets
        self.bilstm = nn.LSTM(config['head_input_dim'], config['lstm_hidden_dim'], config['lstm_layers'], batch_first=True, bidirectional=True, dropout=config.get('dropout', 0.2))
        
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
    def __init__(self, config, targets, label_size):
        super(FFNHead, self).__init__()
        self.targets = targets
        self.label_size = label_size
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
        for hidden_dim in config['fc_hidden_dims']:
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            if config.get('use_batch_norm', False):
                fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(config.get('dropout', 0.2)))
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
