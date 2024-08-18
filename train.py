import torch
from dataloader import ProteinDataset, pad_collate_fn
from torch.utils.data import DataLoader
from model import UNetDenoisingModel, ProteinDiffusion
from evaluation import compute_rmsd
import torch.nn.functional as F

def main():
    pdb_dir = 'data/train'
    val_dir = 'data/val'
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = ProteinDataset(pdb_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    
    val_dataset = ProteinDataset(val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = UNetDenoisingModel().to(device)
    diffusion = ProteinDiffusion(device=device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.num_steps, (batch.shape[0],), device=device)
            x_t, noise = diffusion.q_sample(batch, t)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_rmsd = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                t = torch.randint(0, diffusion.num_steps, (batch.shape[0],), device=device)
                x_t, noise = diffusion.q_sample(batch, t)
                predicted_noise = model(x_t, t)
                val_loss += F.mse_loss(predicted_noise, noise).item()
                val_rmsd += compute_rmsd(batch.cpu().numpy(), x_t.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_rmsd /= len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSD: {val_rmsd:.4f}")

if __name__ == "__main__":
    main()