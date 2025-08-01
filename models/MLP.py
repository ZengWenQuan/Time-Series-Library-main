
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Calculate the input dimension
        # The input will be flattened, so we need to calculate the total number of features.
        # Each of the two feature sequences has a length of seq_len.
        # The number of features per timestep is the second dimension of the input tensor.
        # Let's assume the number of features per timestep is configs.enc_in * 2 (for both sequences)
        input_dim = self.seq_len * configs.enc_in #  for the two feature sequences
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.pred_len * 4)  # 4 for the four target variables
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, num_features]
        # Flatten the input
        x_enc = x_enc.view(x_enc.size(0), -1)
        
        # Pass through the MLP
        output = self.mlp(x_enc)
        
        # Reshape the output to the desired prediction shape
        output = output.view(output.size(0), self.pred_len, 4)
        
        return output
