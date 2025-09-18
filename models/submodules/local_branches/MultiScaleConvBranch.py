import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registries import register_normalized_branch

class MultiScaleConvBlock(nn.Module):
    """
    A block that applies multiple parallel 1D convolutions and a residual connection.
    """
    def __init__(self, input_dim, output_dim, kernel_sizes, dropout):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=k, padding='same') 
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(output_dim * len(kernel_sizes), output_dim, 1)
        self.residual_conv = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        residual = x
        
        conv_outputs = []
        for conv in self.convs:
            out = F.relu(conv(x))
            conv_outputs.append(out)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        projected = self.projection(self.dropout(concatenated))
        
        if self.residual_conv:
            residual = self.residual_conv(residual)
            
        return self.norm(projected + residual)

@register_normalized_branch
class MultiScaleConvBranch(nn.Module):
    """
    A branch that stacks multiple MultiScaleConvBlocks to capture features at various scales.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary containing the configuration for the branch.
        """
        super(MultiScaleConvBranch, self).__init__()

        input_dim = config.get('input_dim', 1)
        num_layers = config.get('num_layers', 2)
        layer_configs = config.get('layer_configs', [
            {'output_dim': 16, 'kernel_sizes': [3, 5, 7]},
            {'output_dim': 32, 'kernel_sizes': [3, 5, 7]}
        ])
        dropout = config.get('dropout', 0.1)

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for i in range(num_layers):
            layer_conf = layer_configs[i]
            output_dim = layer_conf['output_dim']
            kernel_sizes = layer_conf['kernel_sizes']
            self.layers.append(MultiScaleConvBlock(current_dim, output_dim, kernel_sizes, dropout))
            current_dim = output_dim

        self.output_channels = current_dim
        self.output_length = config['input_len']

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input sequence, shape: [batch_size, channels, seq_len].
        """
        for layer in self.layers:
            x = layer(x)
        return x
