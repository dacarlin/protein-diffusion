import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


num_steps = 200


def sample_protein(model, diffusion, length, device):
    model.eval()
    x = torch.randn(1, length, 3, device=device)
    for t in reversed(range(diffusion.num_steps)):
        x = diffusion.p_sample(model, x, t)
    return x.squeeze(0)


class ProteinDiffusion:
    def __init__(
        self, num_steps=num_steps, beta_start=0.01, beta_end=0.07, device="cpu"
    ):
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

    def q_sample(self, x_0, t):
        # Ensure the time tensor 't' is on the same device as x_0
        t = t.to(x_0.device)

        noise = torch.randn_like(x_0, device=x_0.device)
        return (
            torch.sqrt(self.alpha_bar[t])[:, None, None] * x_0
            + torch.sqrt(1 - self.alpha_bar[t])[:, None, None] * noise
        ), noise

    def p_sample(self, model, x_t, t):
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        noise_pred = model(x_t, t_tensor)
        alpha_t = self.alpha[t].to(x_t.device)
        alpha_bar_t = self.alpha_bar[t].to(x_t.device)
        beta_t = self.beta[t].to(x_t.device)

        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )
        var = beta_t * torch.ones_like(x_t)

        return mean + torch.sqrt(var) * torch.randn_like(x_t)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.block(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class DenoisingDiffusionModel(nn.Module):
    def __init__(self, dim=128, n_blocks=4):
        super(DenoisingDiffusionModel, self).__init__()
        self.input_layer = nn.Linear(3, dim)
        self.positional_encoding = PositionalEncoding(dim)
        self.time_encoding = nn.Embedding(num_steps, dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(dim) for _ in range(n_blocks)]
        )
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


class RegularTransformer(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=64, num_heads=8, num_layers=3, num_steps=num_steps
    ):
        super(RegularTransformer, self).__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.time_embedding = nn.Linear(1, hidden_dim)
        self.input_projection = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                RegularTransformerLayer(
                    input_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads
                )
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # Embed time
        t = t.float().view(-1, 1) / float(self.num_steps)  # Normalize time steps
        t_embed = F.relu(self.time_embedding(t)).unsqueeze(1).expand(-1, x.shape[1], -1)

        # Concatenate time embedding with input and project to hidden_dim
        x = torch.cat([x, t_embed], dim=-1)
        h = F.relu(self.input_projection(x))

        for layer in self.layers:
            h = layer(h)

        out = self.fc_out(h)
        return out


class RegularTransformerLayer(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_heads=8):
        super(RegularTransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.qkv_proj = nn.Linear(input_dim, 3 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.fc_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # Compute queries, keys, values
        qkv = self.qkv_proj(x).view(x.shape[0], x.shape[1], 3, self.hidden_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(q, k, v)
        attn_output = self.layer_norm1(x + attn_output)

        # Apply feed-forward network
        ffn_output = self.fc_ffn(attn_output)
        output = self.layer_norm2(attn_output + ffn_output)

        return output


class TensorProductLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TensorProductLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # self.weight shape: (input_dim, output_dim)
        # The einsum operation should match the dimensions correctly
        # and this will error if there aren't enough dimensions 
        batch_size, seq_len, input_dim = x.shape

        # Batched version of multiple linear layers, where each 'i' slice 
        # of the input gets its own linear transformation
        output = torch.einsum("bik,ko->bio", x, self.weight) + self.bias

        return output


class EquivariantAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(EquivariantAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, geometric_features):
        # Compute queries, keys, values
        qkv = self.qkv_proj(x).view(x.shape[0], x.shape[1], 3, self.hidden_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Use 3D Euclidean distance in attention 
        distances = torch.cdist(geometric_features, geometric_features)
        attention_weights = torch.einsum("bik,bjk->bij", q, k) / torch.sqrt(
            torch.tensor(self.hidden_dim)
        )
        attention_weights = attention_weights * torch.exp(-distances)  # Geometric bias

        # Compute attention output
        attention_output = torch.einsum(
            "bij,bjk->bik", F.softmax(attention_weights, dim=-1), v
        )
        return self.fc_out(attention_output)


class EquivariantFeedforward(nn.Module):
    def __init__(self, hidden_dim):
        super(EquivariantFeedforward, self).__init__()
        self.fc1 = TensorProductLayer(hidden_dim, 4 * hidden_dim)
        self.fc2 = TensorProductLayer(4 * hidden_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x))))


class SE3TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SE3TransformerLayer, self).__init__()
        self.attention = EquivariantAttention(hidden_dim, num_heads)
        self.ffn = EquivariantFeedforward(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, geometric_features):
        # Attention
        attn_output = self.attention(x, geometric_features)
        x = self.layer_norm1(x + attn_output)

        # Feedforward
        ffn_output = self.ffn(x)
        return self.layer_norm2(x + ffn_output)


class SE3Transformer(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=64, num_heads=8, num_layers=3, num_steps=num_steps
    ):
        super(SE3Transformer, self).__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.time_embedding = nn.Linear(1, hidden_dim)
        self.input_projection = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [
                SE3TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        # Embed time
        t = t.float().view(-1, 1) / float(self.num_steps)  # Normalize time steps
        t_embed = F.relu(self.time_embedding(t)).unsqueeze(1).expand(-1, x.shape[1], -1)

        # Concatenate time embedding with input and project to hidden_dim
        x = torch.cat([x, t_embed], dim=-1)
        h = F.relu(self.input_projection(x))

        # Extract geometric features (e.g., distances between points)
        geometric_features = torch.cdist(
            h[:, :, :3], h[:, :, :3]
        )  # Using first 3 dims as coordinates

        for layer in self.layers:
            h = layer(h, geometric_features)

        out = self.fc_out(h)
        return out
