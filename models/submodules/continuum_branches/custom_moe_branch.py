
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_continuum_branch
from ..normalized_branches.multiscale_pyramid_branch import MultiScalePyramidBranch

class GatingNetwork(nn.Module):
    """
    门控网络: 分析频域特征，决定专家权重。
    """
    def __init__(self, input_dim, num_experts, k, hidden_dim):
        super(GatingNetwork, self).__init__()
        self.k = k
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x):
        # 使用平均池化来获得一个全局的频域描述符
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
        
        # 1. 初始化门控网络
        fft_output_dim = self.fft_params['n_fft'] // 2 + 1
        self.gating_network = GatingNetwork(
            input_dim=fft_output_dim,
            num_experts=self.moe_params['num_experts'],
            k=self.moe_params['k'],
            hidden_dim=self.moe_params['gating_hidden_dim']
        )
        
        # 2. 初始化专家网络列表
        # 每个专家都是一个多尺度金字塔
        expert_config = config['expert_config']
        expert_config['input_len'] = config['input_len'] # 将父级config的input_len注入到专家config中
        self.experts = nn.ModuleList([
            MultiScalePyramidBranch(expert_config)
            for _ in range(self.moe_params['num_experts'])
        ])
        
        # 3. 确定输出维度
        self.output_dim = self.experts[0].output_dim
        self.output_shape_tuple = self.experts[0].output_shape_tuple

    def forward(self, x):
        # --- 门控逻辑 (在频域) ---
        # 计算STFT以用于门控网络
        with torch.no_grad(): # 通常门控的STFT不需要梯度
            x_fft_r = torch.stft(x, n_fft=self.fft_params['n_fft'], hop_length=self.fft_params['hop_length'], return_complex=False, window=torch.hann_window(self.fft_params['n_fft']).to(x.device))
            x_fft_mag = torch.sqrt(x_fft_r[..., 0]**2 + x_fft_r[..., 1]**2)
        
        # 得到专家的权重和索引
        weights, indices = self.gating_network(x_fft_mag)
        
        # --- 专家处理 (在时域) ---
        # 将原始时域信号 x 传递给所有专家
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # --- 结果融合 ---
        batch_size = x.size(0)
        # 创建一个零张量用于存放最终输出
        final_output = torch.zeros_like(expert_outputs[0])
        
        # 遍历每个样本，根据权重融合专家的输出
        for i in range(batch_size):
            for j in range(self.moe_params['k']):
                expert_idx = indices[i, j]
                weight = weights[i, j]
                final_output[i] += weight * expert_outputs[expert_idx, i]
                
        return final_output
