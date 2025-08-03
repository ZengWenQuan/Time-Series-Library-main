import torch
import torch.nn as nn
import torch.nn.init as init
import yaml

class MPBDBlock(nn.Module):
    """
    Multi-Path Block with Dual branches
    """
    def __init__(self, input_channel=1, output_channel=4, batch_norm=False):
        super(MPBDBlock, self).__init__()
        
        # First branch with smaller kernel sizes
        block1_layers = [
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=not batch_norm),
            nn.ReLU(inplace=True)
        ]
        if batch_norm: block1_layers.insert(1, nn.BatchNorm1d(output_channel))

        self.Block1 = nn.Sequential(*block1_layers)
        
        # Second branch with larger kernel sizes
        block2_layers = [
            nn.Conv1d(input_channel, output_channel, kernel_size=5, stride=1, padding=2, bias=not batch_norm),
            nn.ReLU(inplace=True)
        ]
        if batch_norm: block2_layers.insert(1, nn.BatchNorm1d(output_channel))
        
        self.Block2 = nn.Sequential(*block2_layers)
        
        # Downsample path if input and output channels differ
        self.downsample = nn.Sequential()
        if input_channel != output_channel:
            layers = [nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=not batch_norm)]
            if batch_norm: layers.append(nn.BatchNorm1d(output_channel))
            self.downsample = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.Block1(x) + self.Block2(x) + self.downsample(x)

class Model(nn.Module):
    """
    Multi-Path Block with Dual branches Network, adapted for spectral prediction.
    Configurable via a YAML file.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Load model-specific configuration
        if not hasattr(configs, 'model_conf') or not configs.model_conf:
            raise ValueError("MPBDNet requires a model configuration file specified via --model_conf")

        print(f"Loading model configuration from: {configs.model_conf}")
        with open(configs.model_conf, 'r') as f:
            model_config = yaml.safe_load(f)

        # --- Get parameters from config --- #
        self.input_dim = configs.feature_size
        self.output_dim = configs.label_size
        self.dropout_rate = configs.dropout
        
        batch_norm = model_config.get('batch_norm', False)
        list_inplanes = model_config['list_inplanes']
        embedding_c = model_config['embedding_c']
        rnn_hidden_size = model_config['rnn_hidden_size']
        fc_hidden_dims = model_config['fc_hidden_dims']

        # --- Build the Network --- #
        
        # 1. Convolutional Blocks (MPBD)
        mpbd_layers = []
        current_channels = 1 # Input is a single channel spectrum
        for i, out_channels in enumerate(list_inplanes):
            mpbd_layers.append(MPBDBlock(current_channels, out_channels, batch_norm=batch_norm))
            mpbd_layers.append(nn.AvgPool1d(kernel_size=3, stride=3))
            current_channels = out_channels
        self.mpbd_blocks = nn.Sequential(*mpbd_layers)

        # 2. Calculate the sequence length after convolutions
        seq_len_after_conv = self.input_dim
        for _ in range(len(list_inplanes)):
            seq_len_after_conv = seq_len_after_conv // 3
        
        if seq_len_after_conv <= 0:
            raise ValueError(f"The sequence length becomes non-positive ({seq_len_after_conv}) after convolutional blocks. Try reducing the number of blocks in `list_inplanes` or the pooling kernel/stride.")

        # 3. Recurrent Layers (LSTM)
        self.rnn = nn.LSTM(
            input_size=current_channels, 
            hidden_size=rnn_hidden_size, 
            num_layers=2, # Using 2 layers for more depth
            batch_first=True, 
            bidirectional=True, # Bidirectional is powerful for spectral data
            dropout=self.dropout_rate if self.dropout_rate > 0 else 0
        )
        
        # 4. Fully-Connected Head
        fc_layers = []
        # Input to FC is twice the RNN hidden size because it's bidirectional
        fc_input_dim = seq_len_after_conv * rnn_hidden_size * 2 
        
        for hidden_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(fc_input_dim, hidden_dim))
            if batch_norm: fc_layers.append(nn.BatchNorm1d(hidden_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(self.dropout_rate))
            fc_input_dim = hidden_dim

        fc_layers.append(nn.Linear(fc_input_dim, self.output_dim))
        self.fc_head = nn.Sequential(*fc_layers)

        self._initialize_weights()
        
        print("MPBDNet Model Initialized successfully.")

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None: init.constant_(m.bias, 0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # Input x_enc is expected to have shape [Batch, Seq_Len]
        # We only use the continuum part, which is the first half of the features
        x = x_enc[:, :self.input_dim]
        
        # Add channel dimension: [Batch, Seq_Len] -> [Batch, 1, Seq_Len]
        x = x.unsqueeze(1)
        
        # 1. Pass through convolutional blocks
        x = self.mpbd_blocks(x)
        
        # 2. Prepare for RNN: [Batch, Channels, Seq_Len] -> [Batch, Seq_Len, Channels]
        x = x.permute(0, 2, 1)
        
        # 3. Pass through LSTM
        x, _ = self.rnn(x)
        
        # 4. Flatten and pass through the fully-connected head
        x = x.flatten(start_dim=1)
        x = self.fc_head(x)
        
        return x
