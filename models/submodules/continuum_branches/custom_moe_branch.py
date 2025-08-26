
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_continuum_branch
from ..normalized_branches.multiscale_pyramid_branch import MultiScalePyramidBranch

class GatingNetwork(nn.Module):
    """
    门控网络: 分析频域特征，决定专家权重。
    现在支持通过列表配置多层隐藏层。
    """
    def __init__(self, config):
        super(GatingNetwork, self).__init__()
        self.k = config['k']
        input_dim = config['input_dim']
        num_experts = config['num_experts']
        hidden_dims = config.get('gating_hidden_dims', config.get('gating_hidden_dim'))

        layers = []
        current_dim = input_dim
        
        if isinstance(hidden_dims, list):
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                current_dim = h_dim
        else:
            layers.append(nn.Linear(current_dim, hidden_dims))
            layers.append(nn.ReLU())
            current_dim = hidden_dims
            
        layers.append(nn.Linear(current_dim, num_experts))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        x_pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        logits = self.gate(x_pooled)
        top_k_weights, top_k_indices = torch.topk(logits, self.k, dim=1)
        return F.softmax(top_k_weights, dim=1), top_k_indices

@register_continuum_branch
class CustomMoEBranch(nn.Module):
    """
    自定义混合专家（MoE）分支。
    - 门控网络作用于频域（STFT结果）。
    - 专家网络（MultiScalePyramidBranch）作用于时域（原始连续谱）。
    """
    def __init__(self, config):
        super(CustomMoEBranch, self).__init__()
        self.fft_params = config['fft']
        self.moe_params = config['moe']
        
        # 1. 初始化门控网络 (遵循单一config原则)
        # 将动态计算的 input_dim 添加到 moe_params 中
        self.moe_params['input_dim'] = self.fft_params['n_fft'] // 2 + 1
        self.gating_network = GatingNetwork(self.moe_params)
        
        # 2. 初始化专家网络列表
        expert_config = config['expert_config']
        self.experts = nn.ModuleList([
            MultiScalePyramidBranch(expert_config)
            for _ in range(self.moe_params['num_experts'])
        ])
        
        # 3. 确定输出维度
        self.output_channels = self.experts[0].output_channels
        self.output_len = self.experts[0].output_length

    def forward(self, x):
        # --- 门控逻辑 (在频域) ---
        with torch.no_grad():
            x_fft_r = torch.stft(x, n_fft=self.fft_params['n_fft'], hop_length=self.fft_params['hop_length'], return_complex=False, window=torch.hann_window(self.fft_params['n_fft']).to(x.device))
            x_fft_mag = torch.sqrt(x_fft_r[..., 0]**2 + x_fft_r[..., 1]**2)
        
        weights, indices = self.gating_network(x_fft_mag)
        
        # --- 稀疏专家处理 ---
        batch_size = x.size(0)
        # 从预存的元数据中获取输出形状
        output_shape = (batch_size, self.output_channels, self.output_len)
        final_output = torch.zeros(output_shape, device=x.device)

        # 遍历批次中的每个样本
        for i in range(batch_size):
            # 获取当前样本的输入和其top-k专家的权重与索引
            input_i = x[i].unsqueeze(0)
            top_k_indices_i = indices[i]
            top_k_weights_i = weights[i]

            # 只对选中的专家进行计算并加权求和
            for j in range(self.moe_params['k']):
                expert_idx = top_k_indices_i[j]
                weight = top_k_weights_i[j]
                expert = self.experts[expert_idx]
                final_output[i] += weight * expert(input_i).squeeze(0)
                
        return final_output
