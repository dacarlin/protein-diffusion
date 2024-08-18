import torch
from dataloader import get_dataloader
from model import UNetDenoisingModel
from diffusion import ProteinDiffusion
from training import train
from evaluation import compute_rmsd

def main():
    pdb_dir = 'data'  # Directory containing your PDB files without extensions
    batch_size = 2
    epochs = 100
    learning_rate = 1e-4

    # Check if MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    dataloader = get_dataloader(pdb_dir, batch_size=batch_size)
    
    # Initialize model and diffusion process
    model = UNetDenoisingModel().to(device)
    diffusion = ProteinDiffusion()
    
    # Train the model
    train(model, diffusion, dataloader, num_epochs=epochs, device=device)

if __name__ == "__main__":
    main()
