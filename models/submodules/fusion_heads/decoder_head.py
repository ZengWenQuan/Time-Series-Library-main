import torch
import torch.nn as nn
from ...registries import register_head

@register_head
class DecoderHead(nn.Module):
    """
    解码器式预测头。

    设计思想:
    借鉴自序列到序列(Seq2Seq)模型。它将上游的特征图视为一个包含了光谱所有信息的
    “上下文向量”。本模块首先将此向量映射为循环神经网络(GRU)的初始隐藏状态，
    然后由GRU逐步“解码”，生成每一个目标参数的预测值。

    优点:
    - 结构上更强大，理论上能建模目标参数之间的潜在依赖关系。
    - 将特征作为“状态”而非“输入”，是一种更灵活的特征利用方式。
    """
    def __init__(self, config):
        super().__init__()
        self.targets = config['targets']
        self.num_targets = len(self.targets)
        dropout_rate = config.get('dropout_rate', 0.2)
        hidden_size = config.get('decoder_hidden_size', 128)
        num_layers = config.get('decoder_layers', 2)

        # --- 1. 特征处理层 ---
        self.pool = nn.AdaptiveAvgPool1d(1)
        # 将池化后的特征投射成GRU隐藏状态需要的维度
        self.feature_projector = nn.LazyLinear(hidden_size * num_layers)

        # --- 2. GRU解码器 ---
        # 创建一个可学习的“起始符”作为解码器的第一个输入
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # --- 3. 输出层 ---
        # 将每一步GRU的输出投射到1个预测值
        self.output_projector = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x 的输入形状: (B, C, L)
        batch_size = x.size(0)

        # 1. 将输入特征图转换为初始隐藏状态 h_0
        pooled_features = self.pool(x).squeeze(-1) # -> (B, C)
        h_0 = self.feature_projector(pooled_features) # -> (B, H * n_layers)
        h_0 = h_0.view(batch_size, self.decoder.num_layers, self.decoder.hidden_size) # -> (B, n_layers, H)
        h_0 = h_0.permute(1, 0, 2).contiguous() # -> (n_layers, B, H)

        # 2. 准备解码器的起始输入
        decoder_input = self.start_token.expand(batch_size, -1, -1)

        # 3. 循环解码，生成每个目标的预测值
        predictions = []
        for _ in range(self.num_targets):
            # output: (B, 1, H), h_0: (n_layers, B, H)
            output, h_0 = self.decoder(decoder_input, h_0)
            
            # 将当前步的输出投射为预测值
            step_prediction = self.output_projector(output) # -> (B, 1, 1)
            predictions.append(step_prediction)
            
            # 将当前步的输出作为下一步的输入 (teacher forcing的变体)
            decoder_input = output

        # 4. 拼接所有预测值
        return torch.cat(predictions, dim=-1).squeeze(1) # -> (B, num_targets)
