import torch
from dataloader import ProteinDataset, pad_collate_fn
from torch.utils.data import DataLoader
from model import UNetDenoisingModel, ProteinDiffusion
from evaluation import compute_rmsd
import torch.nn.functional as F

def main():
    pdb_dir = 'data/train'  # Directory containing your PDB files without extensions
    val_dir = 'data/val'
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    train_dataset = ProteinDataset(pdb_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    
    val_dataset = ProteinDataset(val_dir)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    
    # Initialize model and diffusion process
    model = UNetDenoisingModel().to(device)
    diffusion = ProteinDiffusion(device=device)
    
    # Train the model
    #train(model, diffusion, train_dataloader, val_dataloader, num_epochs=epochs, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_dataloader:
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}, RMSD: {rmsd:.4f}")


if __name__ == "__main__":
    main()
