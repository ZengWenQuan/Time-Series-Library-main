
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ...registries import register_global_branch

try:
    from pytorch_wavelets import DWT1D
except ImportError:
    DWT1D = None

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels_per_path):
        super(InceptionBlock, self).__init__()
        self.path1 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=1, padding='same')
        self.path2 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=3, padding='same')
        self.path3 = nn.Conv1d(in_channels, out_channels_per_path, kernel_size=5, padding='same')
        self.output_channels = out_channels_per_path * 3

    def forward(self, x):
        x1 = F.relu(self.path1(x))
        x2 = F.relu(self.path2(x))
        x3 = F.relu(self.path3(x))
        return torch.cat([x1, x2, x3], dim=1)

@register_global_branch
class ContinuumWaveletBranch(nn.Module):
    def __init__(self, config):
        super(ContinuumWaveletBranch, self).__init__()
        if DWT1D is None: 
            raise ImportError("ContinuumWaveletBranch requires pytorch-wavelets. Run: pip install pytorch-wavelets")
        
        # Get global settings
        use_batch_norm = config.get('use_batch_norm', True)
        
        # DWT Layer
        wavelet_levels = config['wavelet_levels']
        self.dwt = DWT1D(wave=config['wavelet_name'], J=wavelet_levels, mode='symmetric')
        
        # CNN Layers
        cnn_layers = []
        in_channels = 1
        
        # Calculate length after DWT
        # Note: This is an approximation that works for 'symmetric' padding and stride=2 DWT.
        current_len = config['input_len'] // (2**wavelet_levels)

        for layer_conf in config['cnn']['layers']:
            block = InceptionBlock(in_channels, layer_conf['out_channels_per_path'])
            cnn_layers.append(block)
            if use_batch_norm: 
                cnn_layers.append(nn.BatchNorm1d(block.output_channels))
            
            pool_size = layer_conf['pool_size']
            cnn_layers.append(nn.MaxPool1d(kernel_size=pool_size))
            
            in_channels = block.output_channels
            current_len = current_len // pool_size
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Set output shape attributes
        self.output_channels = in_channels
        self.output_length = current_len
        self.output_shape_tuple = (self.output_channels, self.output_length)

    def forward(self, x):
        # DWT returns (low_pass, [high_pass_level_1, ...])
        coeffs_low, _ = self.dwt(x)
        
        features = self.cnn(coeffs_low)
        return features
