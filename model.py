import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class DenoisingDiffusionModel(nn.Module):
    def __init__(self, dim=128, n_blocks=4):
        super(DenoisingDiffusionModel, self).__init__()
        self.input_layer = nn.Linear(3, dim)
        self.positional_encoding = PositionalEncoding(dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(n_blocks)])
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.output_layer = nn.Linear(dim, 3)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.positional_encoding(x)
        for block in self.residual_blocks:
            x = block(x)
        x, _ = self.attention(x, x, x)
        x = self.output_layer(x)
        return x
