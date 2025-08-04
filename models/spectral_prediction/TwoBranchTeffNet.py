import torch
import torch.nn as nn
import yaml
import math
from exp.exp_basic import register_model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpectralEncoder(nn.Module):
    """ A CNN followed by a Transformer to encode a spectrum. """
    def __init__(self, model_config, dropout_rate):
        super().__init__()
        patch_conf = model_config['patch_conv']
        trans_conf = model_config['transformer']

        # 1. CNN for patching
        self.patcher = nn.Conv1d(
            in_channels=1,
            out_channels=patch_conf['out_channels'],
            kernel_size=patch_conf['kernel_size'],
            stride=patch_conf['stride']
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(trans_conf['d_model'])
        self.norm = nn.LayerNorm(trans_conf['d_model'])
        self.dropout = nn.Dropout(dropout_rate)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_conf['d_model'],
            nhead=trans_conf['n_heads'],
            dim_feedforward=trans_conf['ffn_dim'],
            dropout=trans_conf['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=trans_conf['num_layers'])

    def forward(self, x):
        # x: [Batch, SeqLen]
        x = x.unsqueeze(1)  # -> [Batch, 1, SeqLen]
        x = self.patcher(x) # -> [Batch, d_model, PatchedLen]
        x = x.permute(0, 2, 1) # -> [Batch, PatchedLen, d_model]
        
        # Add positional encoding, then normalize and apply dropout
        x_pos = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.norm(x_pos)
        x = self.dropout(x)
        
        # Pass through the main transformer encoder
        x = self.transformer_encoder(x)
        return x # [Batch, PatchedLen, d_model]

@register_model('TwoBranchTeffNet')
class TwoBranchTeffNet(nn.Module):
    def __init__(self, configs):
        super(TwoBranchTeffNet, self).__init__()
        self.task_name = configs.task_name
        if not hasattr(configs, 'model_conf') or not configs.model_conf:
            raise ValueError("TwoBranchTeffNet requires a model configuration file.")

        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        self.feature_size = configs.feature_size
        self.output_dim = configs.label_size
        if self.output_dim != 4:
            raise ValueError("This model is designed for an output of 4 labels (Teff + 3 others).")

        # --- Create Shared and Branch-Specific Modules ---
        self.continuum_encoder = SpectralEncoder(model_config, configs.dropout)
        self.normalized_encoder = SpectralEncoder(model_config, configs.dropout)

        d_model = model_config['transformer']['d_model']

        # Branch 1: Teff Prediction Head
        self.teff_head = nn.Linear(d_model, 1)

        # Branch 2: Conditioning and Final Prediction
        self.teff_embedding = nn.Linear(1, d_model) # To embed the predicted Teff
        
        ca_conf = model_config['cross_attention']
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=ca_conf['n_heads'], 
            batch_first=True
        )

        # Final MLP Head for the other 3 parameters
        fc_conf = model_config['final_head_mlp']
        fc_layers = []
        fc_input_dim = d_model
        for hidden_dim in fc_conf['hidden_dims']:
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(fc_conf['dropout']))
            fc_input_dim = hidden_dim
        fc_layers.append(nn.Linear(fc_input_dim, 3)) # Predicts the other 3 labels
        self.final_head = nn.Sequential(*fc_layers)

        print("TwoBranchTeffNet Model Initialized.")

        # Apply custom weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes the weights of the model. """
        if isinstance(module, nn.Linear):
            # Xavier Uniform Initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Initialize weights to 1 and biases to 0 for LayerNorm
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def parameter_estimation(self, x_enc):
        # Split input into continuum (first half) and normalized (second half)
        continuum_spec = x_enc[:, :self.feature_size]
        normalized_spec = x_enc[:, self.feature_size:]

        # --- Branch 1: Predict Teff ---
        continuum_features = self.continuum_encoder(continuum_spec)
        # Use the [CLS] token equivalent - the mean of all patch features
        mean_continuum_features = continuum_features.mean(dim=1)
        pred_teff = torch.tanh(self.teff_head(mean_continuum_features))

        # --- Branch 2: Predict Other Parameters ---
        # 1. Encode the normalized spectrum
        normalized_features = self.normalized_encoder(normalized_spec)

        # 2. Embed the predicted Teff to be used as context
        teff_context = self.teff_embedding(pred_teff).unsqueeze(1) # -> [Batch, 1, d_model]

        # 3. Use cross-attention to condition the normalized features on the Teff context
        # The Teff context "attends to" the normalized features to extract relevant information
        conditioned_features, _ = self.cross_attention(
            query=teff_context,
            key=normalized_features,
            value=normalized_features
        )
        
        # 4. Use the mean of the conditioned features for final prediction
        mean_conditioned_features = conditioned_features.mean(dim=1)
        pred_others = self.final_head(mean_conditioned_features)

        # Concatenate results: [pred_teff (1), pred_others (3)]
        final_output = torch.cat([pred_teff, pred_others], dim=1)
        
        return final_output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'spectral_prediction':
            return self.parameter_estimation(x_enc)
        
        # If other tasks are added in the future, they can be handled here
        # e.g., if self.task_name == 'classification':
        #           return self.classification(x_enc)
        raise "输出有错误"
        return None # Or raise an error for unsupported tasks
