import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

num_steps = 200 

class ProteinDiffusion:
    def __init__(self, num_steps=num_steps, beta_start=0.01, beta_end=0.07, device="cpu"):
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

    def q_sample(self, x_0, t):
        # Ensure the time tensor 't' is on the same device as x_0
        t = t.to(x_0.device)
        
        noise = torch.randn_like(x_0, device=x_0.device)
        return (
            torch.sqrt(self.alpha_bar[t])[:, None, None] * x_0 +
            torch.sqrt(1 - self.alpha_bar[t])[:, None, None] * noise
        ), noise

    def p_sample(self, model, x_t, t):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor)
        alpha_t = self.alpha[t].to(x_t.device)
        alpha_bar_t = self.alpha_bar[t].to(x_t.device)
        beta_t = self.beta[t].to(x_t.device)
        
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        var = beta_t * torch.ones_like(x_t)
        
        return mean + torch.sqrt(var) * torch.randn_like(x_t)
    

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
        self.time_encoding = nn.Embedding(num_steps, dim)
        self.residual_blocks = nn.ModuleList([ResidualBlock(dim) for _ in range(n_blocks)])
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.output_layer = nn.Linear(dim, 3)

    def forward(self, x, t):
        x = self.input_layer(x)
        x = self.positional_encoding(x)
        t = self.time_encoding(t).unsqueeze(1)
        # print(f"{x.shape=} {t.shape=}")
        x = x + t 
        for block in self.residual_blocks:
            x = block(x)
        x, _ = self.attention(x, x, x)
        x = self.output_layer(x)
        return x

