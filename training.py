import torch
import torch.nn.functional as F
from evaluation import compute_rmsd

def train(model, diffusion, dataloader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.num_steps, (batch.shape[0],), device=device)
            x_t, noise = diffusion.q_sample(batch, t)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Compute RMSD on the last batch of the epoch
        rmsd = compute_rmsd(batch.cpu().numpy(), x_t.detach().cpu().numpy())
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, RMSD: {rmsd:.4f}")
