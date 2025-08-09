
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 辅助模块：归一化层获取 ---
def _get_norm_layer(norm_type, num_features):
    if norm_type == 'batchnorm':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layernorm':
        return nn.LayerNorm(num_features)
    elif norm_type in ['none', None]:
        return nn.Identity()
    else:
        raise ValueError(f"未知归一化类型: {norm_type}")

# --- 1. 通道注意力机制 (SE Block) ---
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- 2. 专家模块 (MoE中的卷积金字塔) ---
class SimplePyramidConv(nn.Module):
    def __init__(self, config, norm_type):
        super(SimplePyramidConv, self).__init__()
        layers = []
        for layer_conf in config:
            layers.append(nn.Conv1d(layer_conf['in_channels'], layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], padding=(layer_conf['kernel_size'] - 1) // 2))
            layers.append(_get_norm_layer(norm_type, layer_conf['out_channels']))
            layers.append(nn.ReLU(inplace=True))
            if layer_conf.get('pool_size', 1) > 1:
                layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
        self.pyramid = nn.Sequential(*layers)

    def forward(self, x):
        return self.pyramid(x)

# --- 3. 门控网络 (MoE Gating) ---
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, k, hidden_dim):
        super(GatingNetwork, self).__init__()
        self.k = k
        self.gate = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, x):
        x_pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        logits = self.gate(x_pooled)
        top_k_weights, top_k_indices = torch.topk(logits, self.k, dim=1)
        return F.softmax(top_k_weights, dim=1), top_k_indices

# --- 4. 模型分支 ---
class FrequencyMoEBranch(nn.Module):
    def __init__(self, config, norm_type):
        super(FrequencyMoEBranch, self).__init__()
        self.fft_params = config['fft']
        fft_output_dim = self.fft_params['n_fft'] // 2 + 1
        self.gating_network = GatingNetwork(fft_output_dim, config['moe']['num_experts'], config['moe']['k'], config['moe']['gating_hidden_dim'])
        self.experts = nn.ModuleList([SimplePyramidConv(config['expert_pyramid'], norm_type) for _ in range(config['moe']['num_experts'])])
        # 预计算输出维度
        with torch.no_grad():
            dummy_input = torch.randn(1, fft_output_dim, 100) # 假设一个合理的长度
            self.output_dim = self.experts[0](dummy_input).shape[1]

    def forward(self, x):
        x_fft_r = torch.stft(x, n_fft=self.fft_params['n_fft'], hop_length=self.fft_params['hop_length'], return_complex=False, window=torch.hann_window(self.fft_params['n_fft']).to(x.device))
        x_fft_mag = torch.sqrt(x_fft_r[..., 0]**2 + x_fft_r[..., 1]**2)
        weights, indices = self.gating_network(x_fft_mag)
        batch_size = x.size(0)
        expert_outputs = torch.stack([self.experts[i](x_fft_mag) for i in range(len(self.experts))])
        output = torch.zeros_like(expert_outputs[0])
        for i in range(batch_size):
            for j in range(self.gating_network.k):
                expert_idx = indices[i, j]
                weight = weights[i, j]
                output[i] += weight * expert_outputs[expert_idx, i]
        return output

class LineAttentionBranch(nn.Module):
    def __init__(self, config, norm_type):
        super(LineAttentionBranch, self).__init__()
        layers = []
        in_channels = 1
        for layer_conf in config:
            layers.append(nn.Conv1d(in_channels, layer_conf['out_channels'], kernel_size=layer_conf['kernel_size'], padding=(layer_conf['kernel_size'] - 1) // 2))
            layers.append(_get_norm_layer(norm_type, layer_conf['out_channels']))
            layers.append(nn.ReLU(inplace=True))
            if layer_conf.get('se_reduction', 0) > 0:
                layers.append(SEBlock(layer_conf['out_channels'], reduction=layer_conf['se_reduction']))
            if layer_conf.get('pool_size', 1) > 1:
                layers.append(nn.MaxPool1d(kernel_size=layer_conf['pool_size']))
            in_channels = layer_conf['out_channels']
        self.pyramid = nn.Sequential(*layers)
        self.output_dim = in_channels

    def forward(self, x):
        return self.pyramid(x.unsqueeze(1))

# --- 5. 主模型 ---
@register_model('DualBranchMoENet')
class DualBranchMoENet(nn.Module):
    def __init__(self, configs):
        super(DualBranchMoENet, self).__init__()
        self.task_name = configs.task_name
        self.label_size = configs.label_size

        with open(configs.model_conf, 'r') as f:
            config = yaml.safe_load(f)
        
        # 输入维度由数据加载器和实验脚本处理，模型本身与具体维度解耦
        norm_type = config['normalization_type']

        self.freq_branch = FrequencyMoEBranch(config['freq_branch'], norm_type)
        self.line_branch = LineAttentionBranch(config['line_branch']['pyramid_with_attention'], norm_type)

        fusion_conf = config['fusion_module']
        # 动态计算融合维度
        with torch.no_grad():
            # 使用一个固定长度的、足够大的虚拟张量来计算输出维度
            # 这避免了因输入特征长度过小（如此处为4）而导致卷积核（如大小为5）无法应用的问题
            # 这里的计算只关心通道数，不关心长度，因此使用固定长度是安全的
            DUMMY_LENGTH = 512 
            dummy_continuum = torch.randn(1, DUMMY_LENGTH)
            dummy_normalized = torch.randn(1, DUMMY_LENGTH)

            freq_out_shape = self.freq_branch(dummy_continuum).shape
            line_out_shape = self.line_branch(dummy_normalized).shape
            fusion_input_dim = freq_out_shape[1] + line_out_shape[1]

        self.fusion_lstm = nn.LSTM(fusion_input_dim, fusion_conf['lstm_hidden_dim'], fusion_conf['lstm_layers'], batch_first=True, bidirectional=True, dropout=fusion_conf['dropout_rate'] if fusion_conf['lstm_layers'] > 1 else 0)
        
        ffn_layers = []
        in_dim = fusion_conf['lstm_hidden_dim'] * 2
        for layer_conf in fusion_conf['ffn']:
            ffn_layers.append(nn.Linear(in_dim, layer_conf['out_features']))
            if layer_conf.get('is_output_layer', False) is False:
                ffn_layers.append(nn.ReLU(inplace=True))
                ffn_layers.append(nn.Dropout(fusion_conf['dropout_rate']))
            in_dim = layer_conf['out_features']
        self.prediction_head = nn.Sequential(*ffn_layers)

    def forward(self, x_continuum, x_normalized):
        if self.task_name == 'spectral_prediction':
            return self.regression(x_continuum, x_normalized)
        return x_continuum, x_normalized

    def regression(self, x_continuum, x_normalized):
        # 修正输入维度: [B, L, 1] -> [B, L] 以适配STFT和后续的unsqueeze
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1:
            x_continuum = x_continuum.squeeze(-1)
        if x_normalized.ndim == 3 and x_normalized.shape[2] == 1:
            x_normalized = x_normalized.squeeze(-1)

        freq_features = self.freq_branch(x_continuum)
        line_features = self.line_branch(x_normalized)

        len_freq = freq_features.size(2)
        len_line = line_features.size(2)
        if len_freq != len_line:
            target_len = min(len_freq, len_line)
            freq_features = F.interpolate(freq_features, size=target_len, mode='linear', align_corners=False)
            line_features = F.interpolate(line_features, size=target_len, mode='linear', align_corners=False)

        combined_features = torch.cat([freq_features, line_features], dim=1)
        combined_features = combined_features.permute(0, 2, 1)
        lstm_out, _ = self.fusion_lstm(combined_features)
        sequence_features = lstm_out[:, -1, :]
        predictions = self.prediction_head(sequence_features)
        return predictions
