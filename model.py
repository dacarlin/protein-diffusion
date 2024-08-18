import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinDiffusion:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
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


class UNetDenoisingModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(UNetDenoisingModel, self).__init__()
        self.enc1 = nn.Linear(input_dim + 1, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec1 = nn.Linear(hidden_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t = t.float() / 1000.  # Normalize time steps
        t = t.view(-1, 1).expand(-1, x.shape[1]).unsqueeze(-1)
        x = torch.cat([x, t], dim=-1)
        
        h1 = F.relu(self.enc1(x))
        h2 = F.relu(self.enc2(h1))
        h3 = F.relu(self.dec1(h2 + h1))  # Skip connection
        return self.dec2(h3)
