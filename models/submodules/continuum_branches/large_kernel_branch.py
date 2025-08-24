
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...registries import register_continuum_branch

@register_continuum_branch
class LargeKernelBranch(nn.Module):
    def __init__(self, config):
        super(LargeKernelBranch, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config['stride'],
            padding=(config['kernel_size'] - 1) // 2
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(config['out_channels'], config['fc_dim'])
        self.output_dim = config['fc_dim']

    def forward(self, x):
        x = x.unsqueeze(1) # [B, L] -> [B, 1, L]
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x
