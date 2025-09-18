import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registries import register_block

@register_block
class GlobalAttentionBranch(nn.Module):
    """
    A branch that first patches the input sequence and then uses a Transformer Encoder.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary containing the configuration for the branch.
        """
        super(GlobalAttentionBranch, self).__init__()
        
        patch_size = config.get('patch_size', 16)
        patch_stride = config.get('patch_stride', 10)
        n_heads = config.get('n_heads', 4)
        num_encoder_layers = config.get('num_encoder_layers', 2)
        dim_feedforward = config.get('dim_feedforward', 128)
        dropout = config.get('dropout', 0.1)
        input_len = config['input_length']

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # The embedding dimension is now the patch size
        d_model = patch_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        # --- Calculate output shape attributes ---
        # Calculate padding
        effective_len = input_len - self.patch_size
        pad_size = (self.patch_stride - (effective_len % self.patch_stride)) % self.patch_stride
        padded_len = input_len + pad_size
        
        num_patches = (padded_len - patch_size) // patch_stride + 1
        
        self.output_length = num_patches
        self.output_channels = d_model

    def forward(self, x):
        # x shape: [bs, 1, L]
        
        # Padding
        seq_len = x.shape[2]
        effective_len = seq_len - self.patch_size
        pad_size = (self.patch_stride - (effective_len % self.patch_stride)) % self.patch_stride
        
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))

        # Patching
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_stride)
        # x shape: [bs, 1, num_patches, patch_size]
        
        # Reshape for transformer
        x = x.squeeze(1)
        # x shape: [bs, num_patches, patch_size]

        # Transformer encoder
        output = self.transformer_encoder(x)
        # output shape: [bs, num_patches, patch_size]
        
        # Permute to return as [bs, C, L] which is [bs, patch_size, num_patches]
        return output.permute(0, 2, 1)
