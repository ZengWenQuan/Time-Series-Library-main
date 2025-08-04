#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dual Pyramid Network for Spectral Analysis

Author: Assistant
Date: 2025-08-04

Architecture:
- Two separate PyramidFeatureExtractors for continuum and normalized spectra.
- Feature fusion and a final FFN for regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import yaml
from exp.exp_basic import register_model

class PyramidFeatureExtractor(nn.Module):
    """
    Extracts features from a spectrum using a multi-scale pyramid architecture.
    """
    def __init__(self, configs):
        super(PyramidFeatureExtractor, self).__init__()
        
        self.pyramid_channels = configs.pyramid_channels
        self.use_batch_norm = configs.use_batch_norm
        
        # Input projection layer
        layers = [nn.Conv1d(1, self.pyramid_channels[0], kernel_size=7, padding=3, bias=not self.use_batch_norm)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.pyramid_channels[0]))
        layers.append(nn.ReLU(inplace=True))
        self.input_proj = nn.Sequential(*layers)
        
        # Pyramid blocks sequence
        self.pyramid_blocks = nn.ModuleList()
        input_ch = self.pyramid_channels[0]
        for i in range(len(self.pyramid_channels)):
            output_ch = self.pyramid_channels[i]
            self.pyramid_blocks.append(
                PyramidBlock(
                    input_ch, 
                    output_ch, 
                    kernel_sizes=configs.kernel_sizes, 
                    use_batch_norm=self.use_batch_norm,
                    use_attention=configs.use_attention,
                    attention_reduction=configs.attention_reduction
                )
            )
            input_ch = output_ch * 3 # The output of PyramidBlock concatenates 3 branches

        # Global average pooling to create a fixed-size feature vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.unsqueeze(1) # -> [batch_size, 1, seq_len]
        
        # Input projection
        x = self.input_proj(x)
        
        # Through pyramid blocks
        for block in self.pyramid_blocks:
            x = block(x)
            
        # Global pooling
        x = self.global_pool(x) # -> [batch_size, channels, 1]
        return x.squeeze(-1) # -> [batch_size, channels]

class PyramidBlock(nn.Module):
    """
    A single block in the pyramid, with three parallel branches for different scales.
    """
    def __init__(self, input_channel, output_channel, kernel_sizes, 
                 use_batch_norm, use_attention, attention_reduction):
        super(PyramidBlock, self).__init__()
        
        self.use_attention = use_attention
        
        self.fine_branch = self._make_branch(input_channel, output_channel, kernel_sizes[0], use_batch_norm)
        self.medium_branch = self._make_branch(input_channel, output_channel, kernel_sizes[1], use_batch_norm)
        self.coarse_branch = self._make_branch(input_channel, output_channel, kernel_sizes[2], use_batch_norm)
        
        self.residual = nn.Sequential()
        if input_channel != output_channel * 3:
            layers = [nn.Conv1d(input_channel, output_channel * 3, kernel_size=1, bias=not use_batch_norm)]
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(output_channel * 3))
            self.residual = nn.Sequential(*layers)
            
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(output_channel * 3, max(1, output_channel * 3 // attention_reduction), kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(max(1, output_channel * 3 // attention_reduction), output_channel * 3, kernel_size=1),
                nn.Sigmoid()
            )
        
        self._initialize_weights()
    
    def _make_branch(self, input_channel, output_channel, kernel_size, use_batch_norm):
        padding = kernel_size // 2
        layers = [nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size, padding=padding, bias=not use_batch_norm)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_channel))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        fine_out = self.fine_branch(x)
        medium_out = self.medium_branch(x)
        coarse_out = self.coarse_branch(x)
        
        pyramid_out = torch.cat([fine_out, medium_out, coarse_out], dim=1)
        
        if self.use_attention:
            attention_weights = self.attention(pyramid_out)
            pyramid_out = pyramid_out * attention_weights
        
        return F.relu(pyramid_out + self.residual(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

@register_model('DualPyramidNet')
class DualPyramidNet(nn.Module):
    """
    Dual Pyramid Network for Spectral Analysis.
    """
    def __init__(self, configs):
        super(DualPyramidNet, self).__init__()
        
        self.task_name = configs.task_name
        self.feature_size = configs.feature_size
        self.label_size = configs.label_size

        # Load model-specific config if provided
        if hasattr(configs, 'model_conf') and configs.model_conf:
            with open(configs.model_conf, 'r') as f:
                model_config = yaml.safe_load(f)
            # Overwrite general configs with model-specific ones
            for key, value in model_config.items():
                setattr(configs, key, value)

        # Two separate feature extractors
        self.continuum_extractor = PyramidFeatureExtractor(configs)
        self.normalized_extractor = PyramidFeatureExtractor(configs)
        
        # Calculate the input dimension for the final FFN
        # It's the concatenated output of both extractors
        ffn_input_dim = configs.pyramid_channels[-1] * 3 * 2
        
        # Final Feed-Forward Network
        fc_layers = []
        current_dim = ffn_input_dim
        for hidden_dim in configs.fc_hidden_dims:
            fc_layers.append(nn.Linear(current_dim, hidden_dim))
            fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(configs.dropout))
            current_dim = hidden_dim
        
        fc_layers.append(nn.Linear(current_dim, self.label_size))
        self.ffn = nn.Sequential(*fc_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ['spectral_prediction', 'regression', 'stellar_parameter_estimation']:
            return self.regression(x_enc)
        return None

    def regression(self, x_enc):
        # Split input into continuum and normalized spectra
        continuum_spec = x_enc[:, :self.feature_size]
        normalized_spec = x_enc[:, self.feature_size:]
        
        # Extract features from both parts
        continuum_features = self.continuum_extractor(continuum_spec)
        normalized_features = self.normalized_extractor(normalized_spec)
        
        # Concatenate features
        combined_features = torch.cat([continuum_features, normalized_features], dim=1)
        
        # Final prediction through FFN
        output = self.ffn(combined_features)
        
        return output

if __name__=='main' :
    continuum_data=torch.rand(3,4802)
    normalized_data=torch.rand(3,4802)
    