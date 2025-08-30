import torch
import torch.nn as nn
from ...registries import register_head
from .fusion_modules import FeatureAdjuster

@register_head
class LstmFfnHead(nn.Module):
    """
    一个使用双向LSTM和FFN的预测头。

    1. 使用FeatureAdjuster统一输入维度。
    2. 使用一个双向LSTM处理序列特征。
    3. 拼接LSTM的最终前向和后向隐藏状态。
    4. 使用一个FFN网络从隐藏状态直接回归所有目标值。
    """
    def __init__(self, config):
        super().__init__()
        self.targets = config['targets']
        dropout_rate = config.get('dropout_rate', 0.2)
        in_channels = config['in_channels']
        in_len = config['in_len']

        # --- 1. 入口特征调整器 ---
        self.adjuster = FeatureAdjuster(out_channels=in_channels, out_len=in_len)

        # --- 2. 双向LSTM层 ---
        lstm_hidden_size = config.get('lstm_hidden_size', 128)
        lstm_layers = config.get('lstm_layers', 2)
        self.is_bidirectional = config.get('lstm_bidirectional', True) # 新增：读取双向配置
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=self.is_bidirectional, # 使用配置参数
            dropout=dropout_rate if lstm_layers > 1 else 0
        )

        # --- 3. FFN预测网络 ---
        # 根据是否双向，动态计算FFN的输入维度
        ffn_input_dim = lstm_hidden_size * 2 if self.is_bidirectional else lstm_hidden_size
        ffn_layers_config = config.get('ffn_layers', [128, 64])
        num_targets = len(self.targets)

        layers = []
        current_dim = ffn_input_dim
        for hidden_dim in ffn_layers_config:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_targets))
        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        # x 的输入形状: (B, C_in, L_in)
        
        # 1. 自动调整输入维度
        x = self.adjuster(x) # 输出形状: (B, in_channels, in_len)
        
        # 2. 调整形状以适应LSTM
        x = x.permute(0, 2, 1) # -> (B, L, C)
        
        # 3. 通过LSTM
        # lstm_out 形状: (B, L, hidden_size * 2)
        # h_n 形状: (num_layers * 2, B, hidden_size)
        _ , (h_n, _) = self.lstm(x)
        
        # 4. 提取并拼接最后的隐藏状态
        # h_n[-2] 是前向最后一个隐藏状态, h_n[-1] 是后向最后一个
        last_hidden_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # 5. 通过FFN得到最终预测
        prediction = self.ffn(last_hidden_state)
        
        return prediction
