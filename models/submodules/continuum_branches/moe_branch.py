
import torch
import torch.nn as nn
import torch.nn.functional as F

def _get_norm_layer(norm_type, num_features):
    if norm_type == 'batchnorm': return nn.BatchNorm1d(num_features)
    elif norm_type == 'layernorm': return nn.LayerNorm(num_features)
    else: return nn.Identity()

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

from . import register_continuum_branch

@register_continuum_branch
class FrequencyMoEBranch(nn.Module):
    def __init__(self, config, norm_type):
        super(FrequencyMoEBranch, self).__init__()
        self.fft_params = config['fft']
        fft_output_dim = self.fft_params['n_fft'] // 2 + 1
        self.gating_network = GatingNetwork(fft_output_dim, config['moe']['num_experts'], config['moe']['k'], config['moe']['gating_hidden_dim'])
        self.experts = nn.ModuleList([SimplePyramidConv(config['expert_pyramid'], norm_type) for _ in range(config['moe']['num_experts'])])
        with torch.no_grad():
            dummy_input = torch.randn(1, fft_output_dim, 100)
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
