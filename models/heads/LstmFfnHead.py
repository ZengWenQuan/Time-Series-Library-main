# /home/irving/workspace/Time-Series-Library-main/models/heads/LstmFfnHead.py

import torch
import torch.nn as nn
from models.registries import register_head

@register_head
class LstmFfnHead(nn.Module):
    """为每个目标参数使用独立FFN的LSTM预测头"""
    def __init__(self, cfg):
        super().__init__()
        
        num_targets = len(cfg['targets'])
        lstm_input_dim = cfg['input_channels']
        lstm_hidden_dim = cfg['lstm_hidden_dim']
        num_lstm_layers = cfg['num_lstm_layers']
        ffn_hidden_dim = cfg['ffn_hidden_dim']
        dropout_rate = cfg['dropout']

        # 1. LSTM层保持不变，用于特征提取
        self.lstm = nn.LSTM(
            lstm_input_dim, 
            lstm_hidden_dim, 
            num_lstm_layers, 
            batch_first=True, 
            bidirectional=True
        )

        # 2. 为每个目标参数创建一个独立的FFN头
        ffn_input_dim = lstm_hidden_dim * 2  # 因为LSTM是双向的
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ffn_input_dim, ffn_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ffn_hidden_dim, 1) # 每个头只输出1个值
            )
            for _ in range(num_targets)
        ])

    def forward(self, x):
        # x shape: [B, C, L]
        # 1. 调整维度以适应LSTM
        x_p = x.permute(0, 2, 1) # -> [B, L, C]

        # 2. 通过LSTM提取特征
        # 我们只关心最后一个时间步的输出
        lstm_out, _ = self.lstm(x_p)
        lstm_features = lstm_out[:, -1, :] # -> [B, lstm_hidden_dim * 2]

        # 3. 将LSTM特征分别送入每个独立的FFN头
        outputs = [head(lstm_features) for head in self.heads]

        # 4. 将所有头的输出在通道维度上拼接起来
        # list of [B, 1] -> [B, num_targets]
        return torch.cat(outputs, dim=1)