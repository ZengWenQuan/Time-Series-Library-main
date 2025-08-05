import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 1. 通道注意力机制 (SE Block) ---
class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) 通道注意力模块"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, L]
        Returns:
            torch.Tensor: 经过通道注意力加权后的张量，形状为 [B, C, L]
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- 2. MoE 频率分支 (最终修复版) ---
class Expert(nn.Module):
    """单个CNN专家网络。"""
    def __init__(self, config):
        super(Expert, self).__init__()
        expert_config = config['frequency_branch']['expert_cnn_config']
        use_batch_norm = config['global_settings']['use_batch_norm']
        in_channels = 2
        cnn_layers = []
        for layer_conf in expert_config:
            conv = nn.Conv1d(in_channels, layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], stride=layer_conf['stride'], padding=layer_conf['padding'])
            cnn_layers.append(conv)
            if use_batch_norm:
                cnn_layers.append(nn.BatchNorm1d(layer_conf['out_channels']))
            cnn_layers.append(nn.ReLU(inplace=True))
            in_channels = layer_conf['out_channels']
        self.cnn = nn.Sequential(*cnn_layers)
        self.output_channels = in_channels

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, 2, L_in]
        Returns:
            torch.Tensor: 专家网络的输出，形状为 [B, C_out, L_out]
        """
        return self.cnn(x)

class DeepConvGating(nn.Module):
    """新门控机制：深度卷积门控"""
    def __init__(self, config):
        super(DeepConvGating, self).__init__()
        gating_config = config['frequency_branch']['deep_conv_gating_config']
        num_experts = config['frequency_branch']['moe']['num_experts']
        
        in_channels = 2
        conv_layers = []
        for out_channels in gating_config['conv_out_channels']:
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=gating_config['kernel_size'], padding='same'))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv_net = nn.Sequential(*conv_layers)

        self.final_mlp = nn.Sequential(
            nn.Linear(in_channels, gating_config['mlp_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(gating_config['dropout']),
            nn.Linear(gating_config['mlp_hidden_dim'], num_experts)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, 2, L_fft]
        Returns:
            torch.Tensor: 路由权重，形状为 [B, num_experts]
        """
        # input x: [B, 2, L_fft]
        conv_out = self.conv_net(x)
        pooled_out = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
        return self.final_mlp(pooled_out)

class FrequencyBranchMoE(nn.Module):
    """采用混合专家架构的频率分支 (最终修复版)"""
    def __init__(self, config, device):
        super(FrequencyBranchMoE, self).__init__()
        self.device = device # 存储设备信息
        self.use_windowing = config['frequency_branch']['use_windowing']
        self.window_type = config['frequency_branch'].get('window_type', 'hann')
        self.num_experts = config['frequency_branch']['moe']['num_experts']
        self.top_k = config['frequency_branch']['moe']['top_k']
        self.use_aux_loss = config['frequency_branch']['moe'].get('use_aux_loss', True)
        if self.use_aux_loss:
            self.aux_loss_factor = config['frequency_branch']['moe']['aux_loss_factor']

        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.gating = DeepConvGating(config)
        self.output_channels = self.experts[0].output_channels
        self.output_resizer = nn.AdaptiveMaxPool1d(config['fusion_module']['output_seq_len'])

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, L_seq]
        Returns:
            tuple: (resized_output, aux_loss)
                - resized_output (torch.Tensor): 调整大小后的输出，形状为 [B, C_out, L_fusion]
                - aux_loss (torch.Tensor): 辅助损失，一个标量
        """
        batch_size, seq_len = x.shape

        if self.use_windowing:
            if self.window_type == 'hann':
                window = torch.hann_window(seq_len, device=self.device)
            elif self.window_type == 'hamming':
                window = torch.hamming_window(seq_len, device=self.device)
            else:
                # 默认或不支持的类型，则使用汉宁窗
                window = torch.hann_window(seq_len, device=self.device)
            x = x * window
        
        fft_features = torch.fft.rfft(x, norm='ortho')
        fft_features = torch.stack([fft_features.real, fft_features.imag], dim=1)

        router_logits = self.gating(fft_features)
        routing_weights = F.softmax(router_logits, dim=1)

        if self.use_aux_loss:
            f_i = routing_weights.mean(0)
            p_i = router_logits.mean(0)
            aux_loss = self.aux_loss_factor * self.num_experts * (f_i * p_i).sum()
        else:
            aux_loss = torch.tensor(0.0, device=self.device)

        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)

        # --- Dispatching and Expert Processing ---
        expanded_features = fft_features.repeat_interleave(self.top_k, dim=0)
        flat_top_k_indices = top_k_indices.flatten()

        # Determine the output shape from one expert to correctly initialize buffers
        with torch.no_grad():
            dummy_expert_output = self.experts[0](expanded_features[:1])
            expert_output_channels = dummy_expert_output.shape[1]
            expert_output_len = dummy_expert_output.shape[2]

        # Initialize buffer for all expert outputs
        dispatched_expert_outputs = torch.zeros(
            expanded_features.shape[0],
            expert_output_channels,
            expert_output_len,
            device=expanded_features.device,
            dtype=expanded_features.dtype
        )

        # Loop through experts and apply them to the relevant parts of the batch
        for i in range(self.num_experts):
            mask = (flat_top_k_indices == i)
            if mask.any():
                expert_in = expanded_features[mask]
                expert_out = self.experts[i](expert_in)
                dispatched_expert_outputs[mask] = expert_out
        
        # Weight the outputs and combine
        weighted_expert_outputs = dispatched_expert_outputs * top_k_weights.flatten().view(-1, 1, 1)
        
        # Reshape back and sum over the experts
        final_output = weighted_expert_outputs.view(
            batch_size, 
            self.top_k, 
            expert_output_channels, 
            expert_output_len
        ).sum(dim=1)

        resized_output = self.output_resizer(final_output)
        return resized_output, aux_loss

# --- 其他模块保持不变 ---
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_map, reduction_ratio, use_batch_norm):
        super(InceptionBlock, self).__init__()
        self.paths = nn.ModuleList()
        total_out_channels = 0
        for kernel_size, out_channels in out_channels_map.items():
            layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            self.paths.append(nn.Sequential(*layers))
            total_out_channels += out_channels
        self.attention = ChannelAttention(total_out_channels, reduction_ratio)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C_in, L]
        Returns:
            torch.Tensor: Inception块的输出，形状为 [B, C_out, L]
        """
        path_outputs = [path(x) for path in self.paths]
        concatenated = torch.cat(path_outputs, dim=1)
        attended = self.attention(concatenated)
        return attended

class InceptionBranch(nn.Module):
    def __init__(self, config):
        super(InceptionBranch, self).__init__()
        branch_config = config['inception_branch']
        use_batch_norm = config['global_settings']['use_batch_norm']
        upsample_conf = branch_config['upsampling_conv_transpose']
        self.upsample_layer = nn.Sequential(
            nn.ConvTranspose1d(1, upsample_conf['out_channels'], kernel_size=upsample_conf['kernel_size'], stride=upsample_conf['stride'], padding=upsample_conf['padding']),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList()
        in_channels = upsample_conf['out_channels']
        for block_config in branch_config['blocks']:
            self.blocks.append(InceptionBlock(in_channels, block_config['out_channels_per_path'], block_config['channel_attention_reduction'], use_batch_norm))
            in_channels = sum(block_config['out_channels_per_path'].values())
            self.blocks.append(nn.MaxPool1d(kernel_size=block_config['pool_size']))
        self.output_channels = in_channels
        self.output_resizer = nn.AdaptiveMaxPool1d(config['fusion_module']['output_seq_len'])
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, L_seq]
        Returns:
            torch.Tensor: Inception分支的输出，形状为 [B, C_out, L_fusion]
        """
        x = x.unsqueeze(1)
        x = self.upsample_layer(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_resizer(x)
        return output

class ConvSequenceModule(nn.Module):
    def __init__(self, config, input_channels):
        super(ConvSequenceModule, self).__init__()
        module_config = config['conv_sequence_module']
        use_batch_norm = config['global_settings']['use_batch_norm']
        layers = []
        in_channels = input_channels
        for layer_conf in module_config['conv_layers']:
            layers.append(nn.Conv1d(in_channels, layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], stride=layer_conf['stride'], padding=layer_conf['padding']))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_conf['out_channels']))
            layers.append(nn.ReLU(inplace=True))
            if layer_conf.get('pool_size'):
                layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = layer_conf['out_channels']
        self.conv_net = nn.Sequential(*layers)
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        with torch.no_grad():
            seq_len = config['fusion_module']['output_seq_len']
            dummy_input = torch.randn(1, input_channels, seq_len)
            dummy_output = self.conv_net(dummy_input)
        self.output_dim = dummy_output.shape[1]
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C_in, L_fusion]
        Returns:
            torch.Tensor: 序列模块的输出，形状为 [B, C_out]
        """
        x = self.conv_net(x)
        x = self.final_pool(x)
        return x.squeeze(-1)

class MultiHeadPrediction(nn.Module):
    def __init__(self, config, input_dim, target_names):
        super(MultiHeadPrediction, self).__init__()
        self.heads = nn.ModuleDict()
        head_config = config['prediction_head']['multi_head_config']
        use_batch_norm = config['global_settings']['use_batch_norm']
        dropout_rate = head_config['dropout']
        for target in target_names:
            hidden_dims = head_config['head_hidden_dims'][target]
            layers = []
            in_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout_rate))
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.heads[target] = nn.Sequential(*layers)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C_in]
        Returns:
            torch.Tensor: 多头预测的输出，形状为 [B, num_targets]
        """
        outputs = [self.heads[target](x) for target in self.heads]
        return torch.cat(outputs, dim=-1)

class ConvFFNHead(nn.Module):
    """新的预测头：卷积下采样 + FFN"""
    def __init__(self, config, input_channels, label_size):
        super(ConvFFNHead, self).__init__()
        head_config = config['prediction_head']['conv_ffn_config']
        use_batch_norm = config['global_settings']['use_batch_norm']
        
        # 1. 卷积下采样层
        conv_layers = []
        in_channels = input_channels
        for layer_conf in head_config['conv_layers']:
            conv_layers.append(nn.Conv1d(in_channels, layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], stride=layer_conf['stride'], padding=layer_conf['padding']))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm1d(layer_conf['out_channels']))
            conv_layers.append(nn.ReLU(inplace=True))
            if layer_conf.get('pool_size'):
                conv_layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = layer_conf['out_channels']
        self.conv_net = nn.Sequential(*conv_layers)
        
        # 2. FFN层
        # 动态计算FFN的输入维度
        with torch.no_grad():
            seq_len = config['fusion_module']['output_seq_len']
            dummy_input = torch.randn(1, input_channels, seq_len)
            dummy_output = self.conv_net(dummy_input)
            ffn_input_dim = dummy_output.flatten(1).shape[1]

        ffn_layers = []
        in_dim = ffn_input_dim
        for h_dim in head_config['ffn_hidden_dims']:
            ffn_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                ffn_layers.append(nn.BatchNorm1d(h_dim))
            ffn_layers.append(nn.ReLU(inplace=True))
            ffn_layers.append(nn.Dropout(head_config['dropout']))
            in_dim = h_dim
        ffn_layers.append(nn.Linear(in_dim, label_size))
        self.ffn = nn.Sequential(*ffn_layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C_in, L_fusion]
        Returns:
            torch.Tensor: 预测结果，形状为 [B, label_size]
        """
        x = self.conv_net(x)
        x = x.flatten(1)
        return self.ffn(x)

# --- 主模型 (最终修复版) ---
@register_model('FreqInceptionConvNet')
class FreqInceptionConvNet(nn.Module):
    def __init__(self, configs):
        super(FreqInceptionConvNet, self).__init__()
        self.device = configs.device # 存储设备信息
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size
        self.target_names = configs.targets
        if configs.model_conf is None:
            configs.model_conf = 'conf/freqinceptionconvnet.yaml'
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)
        model_config['feature_size'] = self.feature_size
        self.head_type = model_config['prediction_head']['head_type']
        self.frequency_branch = FrequencyBranchMoE(model_config, self.device)
        self.inception_branch = InceptionBranch(model_config)
        fusion_input_channels = self.frequency_branch.output_channels + self.inception_branch.output_channels
        self.fusion_dropout = nn.Dropout(model_config['fusion_module']['dropout'])
        self.prediction_head = self._build_prediction_head(model_config, fusion_input_channels)
        self._print_model_info(model_config)

    def _build_prediction_head(self, config, fusion_input_channels):
        head_type = config['prediction_head']['head_type']
        if head_type == 'multi_head':
            # 对于multi_head，我们需要先通过ConvSequenceModule提取序列特征
            self.sequence_module = ConvSequenceModule(config, fusion_input_channels)
            return MultiHeadPrediction(config, self.sequence_module.output_dim, self.target_names)
        elif head_type == 'probabilistic':
            self.sequence_module = ConvSequenceModule(config, fusion_input_channels)
            return self._build_probabilistic_head(config, self.sequence_module.output_dim)
        elif head_type == 'conv_ffn':
            # 对于conv_ffn，它自己处理序列，所以我们不需要全局的sequence_module
            self.sequence_module = None
            return ConvFFNHead(config, fusion_input_channels, self.label_size)
        else:
            raise ValueError(f"未知的预测头类型: {head_type}")

    def _build_probabilistic_head(self, config, input_dim):
        head_config = config['prediction_head']['probabilistic_config']
        use_batch_norm = config['global_settings']['use_batch_norm']
        ffn_layers = []
        in_dim = input_dim
        for hidden_dim in head_config['hidden_dims']:
            ffn_layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                ffn_layers.append(nn.BatchNorm1d(hidden_dim))
            ffn_layers.append(nn.ReLU(inplace=True))
            ffn_layers.append(nn.Dropout(head_config['dropout']))
            in_dim = hidden_dim
        ffn_layers.append(nn.Linear(in_dim, self.label_size * 2))
        return nn.Sequential(*ffn_layers)

    def _print_model_info(self, config):
        total_params = sum(p.numel() for p in self.parameters())
        print("=" * 60)
        print(f"FreqInceptionConvNet Model Initialized")
        print(f"  - Prediction Head Type: {self.head_type}")
        print(f"  - Gating Mechanism: DeepConvGating")
        print(f"  - Total Parameters: {total_params:,}")
        print(f"  - Num Experts: {config['frequency_branch']['moe']['num_experts']}, Top-K: {config['frequency_branch']['moe']['top_k']}")
        print("=" * 60)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Args:
            x_enc (torch.Tensor): 输入编码器张量，形状为 [B, L_seq * 2]
        Returns:
            torch.Tensor or tuple: 模型的输出
        """
        if self.task_name == 'spectral_prediction':
            return self.spectral_prediction_forward(x_enc)
        raise ValueError(f"Task name '{self.task_name}' not supported by FreqInceptionConvNet.")

    def spectral_prediction_forward(self, x_enc):
        """
        Args:
            x_enc (torch.Tensor): 输入编码器张量，形状为 [B, L_seq * 2]
        Returns:
            torch.Tensor or tuple:
                - training: (predictions, aux_loss)
                    - predictions (torch.Tensor): 预测结果，形状为 [B, num_targets] 或 [B, num_targets, 2]
                    - aux_loss (torch.Tensor): 辅助损失，一个标量
                - eval: predictions (torch.Tensor): 预测结果
        """
        continuum_spec = x_enc[:, :self.feature_size]
        freq_features, aux_loss = self.frequency_branch(continuum_spec)
        absorption_spec = x_enc[:, self.feature_size:]
        inception_features = self.inception_branch(absorption_spec)
        fused_features = torch.cat((freq_features, inception_features), dim=1)
        fused_features = self.fusion_dropout(fused_features)
        
        if self.sequence_module is not None:
            sequence_features = self.sequence_module(fused_features)
            predictions = self.prediction_head(sequence_features)
        else:
            # 当使用ConvFFNHead时，直接将融合特征传入
            predictions = self.prediction_head(fused_features)
        if self.head_type == 'probabilistic':
            predictions = predictions.view(-1, self.label_size, 2)
        if self.training:
            return predictions, aux_loss
        else:
            return predictions
