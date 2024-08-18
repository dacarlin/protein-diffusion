import torch
import torch.nn as nn
import torch.nn.functional as F

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
