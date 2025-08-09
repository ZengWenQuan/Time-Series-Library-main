
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from exp.exp_basic import register_model

# --- 1. Sub-path Modules for the MSST Branch ---
class CNNPath(nn.Module):
    """Extracts local features using a CNN pyramid."""
    def __init__(self, config):
        super(CNNPath, self).__init__()
        layers = []
        in_channels = 1
        for layer_conf in config:
            layers.append(nn.Conv1d(in_channels, layer_conf['out_channels'], layer_conf['kernel_size'], padding='same'))
            layers.append(nn.BatchNorm1d(layer_conf['out_channels']))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(layer_conf['pool_size']))
            in_channels = layer_conf['out_channels']
        self.cnn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = in_channels

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.pool(x)
        return x.squeeze(-1)

class TransformerPath(nn.Module):
    """Extracts global features using a patch-based Transformer with auto-padding."""
    def __init__(self, feature_size, config):
        super(TransformerPath, self).__init__()
        patch_len = config['patch_len']
        d_model = config['d_model']

        # --- ADDED: Padding Logic ---
        if feature_size % patch_len == 0:
            self.padding_needed = 0
            num_patches = feature_size // patch_len
        else:
            self.padding_needed = patch_len - (feature_size % patch_len)
            num_patches = (feature_size + self.padding_needed) // patch_len
        
        self.patching = nn.Linear(patch_len, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, config['n_head'], config['d_ff'], config['dropout'], batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, config['n_layers'])
        self.output_dim = d_model

    def forward(self, x):
        # --- ADDED: Apply padding if necessary ---
        if self.padding_needed > 0:
            x = F.pad(x, (0, self.padding_needed), "constant", 0)

        patches = x.unfold(dimension=-1, size=self.patching.in_features, step=self.patching.in_features)
        x = self.patching(patches)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        return x[:, 0] # Return the CLS token representation

class FFTPath(nn.Module):
    """Extracts frequency features using FFT and an MLP."""
    def __init__(self, feature_size, config):
        super(FFTPath, self).__init__()
        fft_len = feature_size // 2 + 1
        layers = []
        in_features = fft_len
        for out_features in config['mlp_layers']:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        self.mlp = nn.Sequential(*layers)
        self.output_dim = in_features

    def forward(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho')
        x_fft_mag = x_fft.abs()
        return self.mlp(x_fft_mag)

# --- 2. Main Branches ---
class NormalizedMSSTBranch(nn.Module):
    """The core branch that processes the normalized spectrum with three paths."""
    def __init__(self, feature_size, config):
        super(NormalizedMSSTBranch, self).__init__()
        self.cnn_path = CNNPath(config['cnn_path'])
        self.transformer_path = TransformerPath(feature_size, config['transformer_path'])
        self.fft_path = FFTPath(feature_size, config['fft_path'])
        self.output_dim = self.cnn_path.output_dim + self.transformer_path.output_dim + self.fft_path.output_dim

    def forward(self, x):
        f_cnn = self.cnn_path(x)
        f_transformer = self.transformer_path(x)
        f_fft = self.fft_path(x)
        return torch.cat([f_cnn, f_transformer, f_fft], dim=1)

class ContinuumCNNBranch(CNNPath):
    """A simpler CNN-only branch for the continuum spectrum."""
    def __init__(self, config):
        super(ContinuumCNNBranch, self).__init__(config)

# --- 3. Main Model: MultiScaleSpectralTransformer ---
@register_model('MultiScaleSpectralTransformer')
class MultiScaleSpectralTransformer(nn.Module):
    def __init__(self, configs):
        super(MultiScaleSpectralTransformer, self).__init__()
        self.task_name = configs.task_name
        
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        self.continuum_branch = ContinuumCNNBranch(model_config['continuum_cnn_branch'])
        self.normalized_branch = NormalizedMSSTBranch(configs.feature_size, model_config['normalized_msst_branch'])

        fusion_input_dim = self.continuum_branch.output_dim + self.normalized_branch.output_dim
        head_layers = []
        in_features = fusion_input_dim
        for layer_conf in model_config['fusion_head']:
            head_layers.append(nn.Linear(in_features, layer_conf['out_features']))
            if not layer_conf.get('is_output_layer', False):
                head_layers.append(nn.ReLU(inplace=True))
                head_layers.append(nn.Dropout(layer_conf.get('dropout', 0.2)))
            in_features = layer_conf['out_features']
        self.prediction_head = nn.Sequential(*head_layers)

    def forward(self, x_continuum, x_normalized):
        if x_continuum.ndim == 3 and x_continuum.shape[2] == 1:
            x_continuum = x_continuum.squeeze(-1)
        if x_normalized.ndim == 3 and x_normalized.shape[2] == 1:
            x_normalized = x_normalized.squeeze(-1)

        continuum_features = self.continuum_branch(x_continuum)
        normalized_features = self.normalized_branch(x_normalized)

        fused_features = torch.cat([continuum_features, normalized_features], dim=1)
        predictions = self.prediction_head(fused_features)
        return predictions
