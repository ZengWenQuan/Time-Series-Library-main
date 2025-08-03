import torch.nn as nn
import yaml

class Model(nn.Module):
    """
    A dynamically configurable MLP model for high-dimensional spectral regression tasks.
    The architecture can be defined via a YAML configuration file.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.feature_size * 2
        self.output_dim = configs.label_size
        self.dropout_rate = configs.dropout

        layers = []
        current_dim = self.input_dim

        # Load architecture from YAML file if provided
        if hasattr(configs, 'model_conf') and configs.model_conf:
            print(f"Loading model configuration from: {configs.model_conf}")
            with open(configs.model_conf, 'r') as f:
                model_config = yaml.safe_load(f)
            
            architecture = model_config.get('architecture', [])
            
            for layer_conf in architecture:
                layer_type = layer_conf.get('type')
                if layer_type == 'linear':
                    neurons = layer_conf['neurons']
                    layers.append(nn.Linear(current_dim, neurons))
                    current_dim = neurons
                elif layer_type == 'relu':
                    layers.append(nn.ReLU())
                elif layer_type == 'dropout':
                    # Use the dropout rate from the main config for consistency
                    layers.append(nn.Dropout(self.dropout_rate))
        else:
            # Fallback to a default simple architecture if no config file is provided
            print("No model configuration file provided. Using default MLP architecture.")
            hidden_dim = configs.d_model
            layers.append(nn.Linear(self.input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            current_dim = hidden_dim

        # Add the final output layer
        layers.append(nn.Linear(current_dim, self.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        print(f"MLP Model Initialized:")
        print(f"  - Input Dimensions: {self.input_dim}")
        print(f"  - Output Dimensions: {self.output_dim}")
        # Print the actual network structure
        print("  - Architecture:")
        for name, module in self.network.named_children():
            print(f"    {name}: {module}")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.network(x_enc)
