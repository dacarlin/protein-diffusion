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


class AFBackboneRepresentation(nn.Module):
    def __init__(self):
        super(AFBackboneRepresentation, self).__init__()

    def forward(self, positions):
        """
        Compute the local frames for each residue's backbone.

        Args:
        positions: Tensor of shape (num_residues, 4, 3) containing the coordinates
                   of N, CA, C, O atoms for each residue.

        Returns:
        frames: Tensor of shape (num_residues, 4, 4) containing the rotation matrix
                and translation vector for each residue's local frame.
        """
        N, CA, C = positions[:, 0], positions[:, 1], positions[:, 2]

        # Compute the local coordinate system
        t = CA - N
        x = normalize(C - N)
        z = normalize(torch.cross(t, x))
        y = torch.cross(z, x)

        # Create rotation matrices
        rot_mats = torch.stack([x, y, z], dim=-1)

        # Create translation vectors (CA atom positions)
        trans = CA

        # Combine rotation and translation into 4x4 transformation matrices
        zeros = torch.zeros_like(trans[:, :1])
        ones = torch.ones_like(zeros)

        frames = torch.cat(
            [
                torch.cat([rot_mats, trans.unsqueeze(-1)], dim=-1),
                torch.cat([zeros, zeros, zeros, ones], dim=-1).unsqueeze(1),
            ],
            dim=1,
        )

        return frames


def normalize(v):
    """Normalize a vector."""
    return v / torch.norm(v, dim=-1, keepdim=True)


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


import torch
import torch.nn as nn
import torch.nn.functional as F


class TensorProductLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TensorProductLayer, self).__init__()
        # Initialize weights and biases
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, input_dim)
        # self.weight shape: (input_dim, output_dim)
        # mask shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.shape

        # Perform tensor product operation
        # This is equivalent to applying a different linear transformation to each 'i' slice of the input
        # Gradient flows through this operation to both x and self.weight
        output = torch.einsum("bik,ko->bio", x, self.weight) + self.bias

        return output * mask.unsqueeze(-1)  # Apply mask

    # Visualization suggestion:
    # def visualize_weights(self):
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(self.weight.detach().cpu().numpy(), cmap='viridis')
    #     plt.colorbar()
    #     plt.title('TensorProductLayer Weights')
    #     plt.xlabel('Output Dimension')
    #     plt.ylabel('Input Dimension')
    #     plt.show()


class EquivariantAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(EquivariantAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Linear projections for query, key, and value
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, geometric_features, mask):
        # Compute queries, keys, values
        # Shape: (batch_size, seq_len, 3, hidden_dim)
        qkv = self.qkv_proj(x).view(x.shape[0], x.shape[1], 3, self.hidden_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Compute 3D Euclidean distances between all pairs of points
        # Shape: (batch_size, seq_len, seq_len)
        distances = torch.cdist(geometric_features, geometric_features)

        # Compute attention scores
        # Shape: (batch_size, seq_len, seq_len)
        attention_weights = torch.einsum("bik,bjk->bij", q, k) / torch.sqrt(
            torch.tensor(self.hidden_dim)
        )

        # Apply geometric bias to attention weights
        # This makes the attention mechanism sensitive to the spatial structure of the input
        attention_weights = attention_weights * torch.exp(-distances)  # Geometric bias

        # Apply mask to attention weights
        attention_weights = attention_weights.masked_fill(
            ~mask.unsqueeze(1), float("-inf")
        )

        # Compute attention output
        # Gradient flows through attention_weights to q, k, and through v
        attention_output = torch.einsum(
            "bij,bjk->bik", F.softmax(attention_weights, dim=-1), v
        )

        # Final projection
        return self.fc_out(attention_output) * mask.unsqueeze(-1)  # Apply mask

    # Visualization suggestion:
    # def visualize_attention(self, attention_weights, step):
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(attention_weights[0].detach().cpu().numpy(), cmap='viridis')
    #     plt.colorbar()
    #     plt.title(f'Attention Weights at Step {step}')
    #     plt.xlabel('Key Position')
    #     plt.ylabel('Query Position')
    #     plt.show()


class EquivariantFeedforward(nn.Module):
    def __init__(self, hidden_dim):
        super(EquivariantFeedforward, self).__init__()
        # Two tensor product layers with expansion factor of 4
        self.fc1 = TensorProductLayer(hidden_dim, 4 * hidden_dim)
        self.fc2 = TensorProductLayer(4 * hidden_dim, hidden_dim)

    def forward(self, x, mask):
        # Apply two tensor product layers with ReLU activation
        # Gradient flows through both layers and ReLU operations
        return F.relu(self.fc2(F.relu(self.fc1(x, mask)), mask))


class SE3TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(SE3TransformerLayer, self).__init__()
        self.attention = EquivariantAttention(hidden_dim, num_heads)
        self.ffn = EquivariantFeedforward(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, geometric_features, mask):
        # Attention sub-layer
        attn_output = self.attention(x, geometric_features, mask)
        x = self.layer_norm1(x + attn_output)  # Residual connection and normalization

        # Feedforward sublayer
        ffn_output = self.ffn(x, mask)
        return self.layer_norm2(x + ffn_output)  # Residual connection and norm

    # Visualization suggestion:
    # def visualize_layer_output(self, layer_output, step):
    #     plt.figure(figsize=(15, 5))
    #     plt.imshow(layer_output[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
    #     plt.colorbar()
    #     plt.title(f'SE3TransformerLayer Output at Step {step}')
    #     plt.xlabel('Hidden Dimension')
    #     plt.ylabel('Sequence Position')
    #     plt.show()


class SE3Transformer(nn.Module):
    def __init__(
        self, input_dim=3, hidden_dim=64, num_heads=8, num_layers=3, num_steps=num_steps
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
        # Embed and normalize time
        # Shape: (batch_size, 1)
        t = t.float().view(-1, 1) / float(self.num_steps)
        # Shape: (batch_size, seq_len, hidden_dim)
        t_embed = F.relu(self.time_embedding(t)).unsqueeze(1).expand(-1, x.shape[1], -1)

        # Concatenate time embedding with input and project to hidden_dim
        # Shape: (batch_size, seq_len, input_dim + hidden_dim)
        x = torch.cat([x, t_embed], dim=-1)
        # Shape: (batch_size, seq_len, hidden_dim)
        h = F.relu(self.input_projection(x))

        # Extract geometric features (e.g., distances between points)
        # Shape: (batch_size, seq_len, seq_len)
        geometric_features = torch.cdist(
            h[:, :, :3], h[:, :, :3]
        )  # Using first 3 dims as coordinates

        # Apply SE3TransformerLayers
        for layer in self.layers:
            h = layer(h, geometric_features, mask)

        # Final output projection
        # Shape: (batch_size, seq_len, input_dim)
        out = self.fc_out(h)
        return out * mask.unsqueeze(-1)  # Apply mask to output

    # Visualization suggestion:
    # def visualize_model_output(self, model_output, step):
    #     plt.figure(figsize=(15, 5))
    #     plt.imshow(model_output[0].detach().cpu().numpy(), cmap='viridis', aspect='auto')
    #     plt.colorbar()
    #     plt.title(f'SE3Transformer Output at Step {step}')
    #     plt.xlabel('Output Dimension')
    #     plt.ylabel('Sequence Position')
    #     plt.show()
