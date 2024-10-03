import torch
import torch.nn as nn
import torch.nn.functional as F
import math


num_steps = 1000


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

        # Total number of diffusion steps
        self.num_steps = num_steps
        # Create a linear schedule for beta values
        self.beta = torch.linspace(beta_start, beta_end, num_steps, device=device)
        # Calculate alpha values (1 - beta)
        self.alpha = 1 - self.beta
        # Calculate cumulative product of alpha values
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)

        # plt.plot(self.beta.cpu().numpy(), label='beta')
        # plt.plot(self.alpha.cpu().numpy(), label='alpha')
        # plt.plot(self.alpha_bar.cpu().numpy(), label='alpha_bar')
        # plt.legend()
        # plt.title('Diffusion schedule')
        # plt.xlabel('Step')
        # plt.ylabel('Value')
        # plt.show()

    def q_sample(self, x_0, t):
        # Forward diffusion process (q distribution)
        # x_0: initial protein structure
        # t: timestep

        # Ensure the time tensor 't' is on the same device as x_0
        t = t.to(x_0.device)
        # Generate random noise
        noise = torch.randn_like(x_0, device=x_0.device)

        # Apply the forward diffusion equation:
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        return (
            torch.sqrt(self.alpha_bar[t])[:, None, None] * x_0
            + torch.sqrt(1 - self.alpha_bar[t])[:, None, None] * noise
        ), noise

        # Visualization suggestion:
        # def visualize_diffusion(x_0, x_t, step):
        #     fig = plt.figure(figsize=(12, 5))
        #     ax1 = fig.add_subplot(121, projection='3d')
        #     ax2 = fig.add_subplot(122, projection='3d')
        #     ax1.scatter(x_0[0, :, 0], x_0[0, :, 1], x_0[0, :, 2])
        #     ax2.scatter(x_t[0, :, 0], x_t[0, :, 1], x_t[0, :, 2])
        #     ax1.set_title('Original Structure')
        #     ax2.set_title(f'Diffused Structure (Step {step})')
        #     plt.show()

    def p_sample(self, model, x_t, t):
        # Reverse diffusion process (p distribution)
        # model: the denoising model
        # x_t: noisy protein structure at time t
        # t: current timestep

        # Create a tensor of timesteps
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # Predict the noise using the model
        mask = torch.ones(len(x_t), dtype=torch.bool).to(x_t.device)
        noise_pred = model(x_t, t_tensor, mask)

        # Get alpha and beta values for the current timestep
        alpha_t = self.alpha[t].to(x_t.device)
        alpha_bar_t = self.alpha_bar[t].to(x_t.device)
        beta_t = self.beta[t].to(x_t.device)

        # Calculate the mean of the reverse diffusion process
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )

        # Calculate the variance
        var = beta_t * torch.ones_like(x_t)

        # Sample from the reverse diffusion distribution
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

        # Visualization suggestion:
        # def visualize_reverse_diffusion(x_t, denoised, step):
        #     fig = plt.figure(figsize=(12, 5))
        #     ax1 = fig.add_subplot(121, projection='3d')
        #     ax2 = fig.add_subplot(122, projection='3d')
        #     ax1.scatter(x_t[0, :, 0], x_t[0, :, 1], x_t[0, :, 2])
        #     ax2.scatter(denoised[0, :, 0], denoised[0, :, 1], denoised[0, :, 2])
        #     ax1.set_title(f'Noisy Structure (Step {step})')
        #     ax2.set_title(f'Denoised Structure (Step {step})')
        #     plt.show()


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


class SE3PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=1000):
        super(SE3PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Create a learnable parameter for each relative position
        self.relative_positions = nn.Parameter(torch.randn(2 * max_len - 1, hidden_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        seq_len = x.size(1)

        # Create a range of relative positions
        positions = torch.arange(seq_len, device=x.device).unsqueeze(1) - torch.arange(
            seq_len, device=x.device
        ).unsqueeze(0)
        positions += self.max_len - 1  # Shift to positive indices

        # Get the corresponding encodings
        relative_encodings = self.relative_positions[positions]

        return relative_encodings  # Shape: (seq_len, seq_len, hidden_dim)


class EquivariantAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(EquivariantAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Projections for query, key, value
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

        # Projection for geometric features
        self.geo_proj = nn.Linear(2, num_heads)

        # Projection for positional encodings
        self.pos_proj = nn.Linear(hidden_dim, num_heads)

    def forward(self, x, geometric_features, positional_encodings, mask):
        # x shape: (batch_size, seq_len, hidden_dim)
        # geometric_features shape: (batch_size, seq_len, seq_len, 2)
        # positional_encodings shape: (seq_len, seq_len, hidden_dim)
        # mask shape: (batch_size, seq_len)
        batch_size, seq_len, _ = x.shape

        # Project input to query, key, value
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, -1)
        q, k, v = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )  # Each has shape (batch_size, seq_len, num_heads, head_dim)

        # Project geometric features
        geo_weights = self.geo_proj(geometric_features).permute(
            0, 3, 1, 2
        )  # (batch_size, num_heads, seq_len, seq_len)

        # Project positional encodings
        pos_weights = self.pos_proj(positional_encodings)
        pos_weights = (
            pos_weights.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 3, 1, 2)
        )  # (batch_size, num_heads, seq_len, seq_len)

        # Compute attention scores
        attention_weights = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(
            self.hidden_dim // self.num_heads
        )

        # Apply geometric and positional biases
        attention_weights = attention_weights + geo_weights + pos_weights

        # Apply mask
        attention_weights = attention_weights.masked_fill(
            ~mask.unsqueeze(1).unsqueeze(1), float("-inf")
        )

        # Compute attention output
        attention_output = torch.einsum(
            "bhij,bjhd->bihd", F.softmax(attention_weights, dim=-1), v
        )
        attention_output = attention_output.reshape(
            batch_size, seq_len, self.hidden_dim
        )

        return self.fc_out(attention_output)


class EquivariantFeedforward(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super(EquivariantFeedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        self.fc2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, hidden_dim)
        # mask shape: (batch_size, seq_len)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x * mask.unsqueeze(-1)  # Apply mask


class SE3TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SE3TransformerLayer, self).__init__()
        self.attention = EquivariantAttention(hidden_dim, num_heads)
        self.ffn = EquivariantFeedforward(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, positional_encodings, mask):
        # x shape: (batch_size, seq_len, hidden_dim)
        # positional_encodings shape: (seq_len, seq_len, hidden_dim)
        # mask shape: (batch_size, seq_len)

        batch_size, seq_len, hidden_dim = x.size()

        # Compute pairwise differences for geometric features
        x_i = x.unsqueeze(2)  # (batch_size, seq_len, 1, hidden_dim)
        x_j = x.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_dim)
        diff = x_i - x_j  # (batch_size, seq_len, seq_len, hidden_dim)

        # Euclidean distance (L2 norm)
        distances = torch.norm(diff, dim=-1)  # (batch_size, seq_len, seq_len)

        # Dot product between pairs
        # dot_products = torch.einsum('bijd,bijd->bij', diff, diff)  # (batch_size, seq_len, seq_len)
        dot_products = torch.matmul(diff, diff.transpose(-2, -1)).diagonal(
            dim1=-2, dim2=-1
        )  # (batch_size, seq_len, seq_len)

        # Stack distances and dot products
        geometric_features = torch.stack(
            [distances, dot_products], dim=-1
        )  # (batch_size, seq_len, seq_len, 2)

        # Attention sub-layer
        attn_output = self.attention(x, geometric_features, positional_encodings, mask)
        x = self.layer_norm1(x + attn_output)  # Residual connection and normalization

        # Feedforward sub-layer
        ffn_output = self.ffn(x, mask)
        x = self.layer_norm2(x + ffn_output)  # Residual connection and normalization

        return x


class SE3Transformer(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_dim=64,
        num_heads=4,
        num_layers=3,
        num_steps=num_steps,
        max_len=512,
    ):
        super(SE3Transformer, self).__init__()
        self.num_steps = num_steps
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Time embedding
        self.time_embedding = nn.Linear(1, hidden_dim)

        # Input projection
        self.input_projection = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # SE3 positional encoding
        self.positional_encoding = SE3PositionalEncoding(hidden_dim, max_len)

        # Stack of SE3TransformerLayers
        self.layers = nn.ModuleList(
            [
                SE3TransformerLayer(hidden_dim=hidden_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t, mask):
        # x shape: (batch_size, seq_len, input_dim)
        # t shape: (batch_size,)
        # mask shape: (batch_size, seq_len)

        # Embed and normalize time
        t = t.float().view(-1, 1) / float(self.num_steps)
        t_embed = F.relu(self.time_embedding(t)).unsqueeze(1).expand(-1, x.shape[1], -1)

        # Concatenate time embedding with input and project to hidden_dim
        x = torch.cat([x, t_embed], dim=-1)
        h = F.relu(self.input_projection(x))

        # Compute positional encodings
        positional_encodings = self.positional_encoding(h)

        # Apply SE3TransformerLayers
        for layer in self.layers:
            h = layer(h, positional_encodings, mask)

        # Final output projection
        out = self.fc_out(h)
        return out * mask.unsqueeze(-1)  # Apply mask to output


# Example usage:
# model = SE3Transformer(input_dim=3, hidden_dim=64, num_heads=8, num_layers=3, num_steps=100, feature_dim=16, max_len=1000)
# x = torch.randn(32, 100, 3)  # (batch_size, seq_len, input_dim)
# t = torch.randint(0, 100, (32,))  # (batch_size,)
# mask = torch.ones(32, 100, dtype=torch.bool)  # (batch_size, seq_len)
# output = model(x, t, mask)
