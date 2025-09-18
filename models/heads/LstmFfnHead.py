# /home/irving/workspace/Time-Series-Library-main/models/heads/LstmFfnHead.py

import torch.nn as nn
from models.registries import register_head

@register_head
class LstmFfnHead(nn.Module):
    """简化的LSTM+FFN预测头"""
    def __init__(self, cfg):
        super().__init__()
        self.lstm = nn.LSTM(cfg['input_channels'], cfg['lstm_hidden_dim'], cfg['num_lstm_layers'], batch_first=True, bidirectional=True)
        self.ffn = nn.Sequential(
            nn.Linear(cfg['lstm_hidden_dim'] * 2, cfg['ffn_hidden_dim']),
            nn.ReLU(), nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['ffn_hidden_dim'], len(cfg['targets']))
        )

    def forward(self, x):
        x_p = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x_p)
        return self.ffn(lstm_out[:, -1, :])