import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Critic(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, num_heads, dropout=0.2):
        super(Critic, self).__init__()

        self.input_linear = nn.Linear(5, hidden_size)

        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4,
                                                dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.output_linear(out)
        return out
