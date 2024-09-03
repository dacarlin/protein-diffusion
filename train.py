import torch
from dataloader import ProteinDataset, pad_collate_fn
from torch.utils.data import DataLoader
from model import (
    ProteinDiffusion,
    SE3Transformer,
    RegularTransformer,
    sample_protein,
)
from evaluation import compute_rmsd
import torch.nn.functional as F
import os
import numpy as np
from random import seed, shuffle
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


torch.manual_seed(99)
seed(99)


def visualize_batch(original, noisy, step):
    original = original.cpu()
    noisy = noisy.cpu()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={"projection": "3d"})
    for i in range(4):
        axes[0, i].scatter(original[i, :, 0], original[i, :, 1], original[i, :, 2])
        axes[0, i].set_title(f"Original Structure {i+1}")
        axes[1, i].scatter(noisy[i, :, 0], noisy[i, :, 1], noisy[i, :, 2])
        axes[1, i].set_title(f"Noisy Structure {i+1}")
    plt.suptitle(f"Batch Visualization at Step {step}")
    plt.tight_layout()
    plt.savefig(f"samples/batch_vis_step_{step}.png")
    plt.close()


def xyz_to_pdb(xyz_tensor, pdb_file):
    """Create a poly-Gly peptide PDB file from a tensor of CA coordiates (L, 3)"""

    with open(pdb_file, 'w') as f:
        xyz = xyz_tensor.tolist()
        for i, line in enumerate(xyz, start=1):
            x, y, z = line 
            f.write(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")


def generate_and_save_sample(model, diffusion, max_protein_length, step, device):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, max_protein_length, 3).to(device)
        for t in reversed(range(diffusion.num_steps)):
            x = diffusion.p_sample(model, x, t)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[0, :, 0].cpu(), x[0, :, 1].cpu(), x[0, :, 2].cpu())
    ax.set_title(f"Generated Protein Structure at Step {step}")
    plt.savefig(f"samples/generated_step_{step}.png")
    plt.close()
    #xyz_to_pdb(x, "samples/generated_step_{step}.pdb")
    model.train()


def main():
    # run params
    pdb_dir = "data/dompdb"
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    sample_length = 200  # length of protein to sample
    sample_interval = 20
    visualization_interval = 20
    max_samples = 45_000
    fit_one_sample = False

    # create device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # load and reproducibly shuffle datasets
    pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir)][:max_samples]
    shuffle(pdb_files)
    n1 = int(0.9 * len(pdb_files))
    train_files = pdb_files[:n1]
    val_files = pdb_files[n1:]
    if fit_one_sample:
        train_files = pdb_files[0:1]
        val_files = pdb_files[0:1]
    print(f"train_samples={len(train_files)} val_samples={len(val_files)}")

    # create dataloaders
    train_dataset = ProteinDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_dataset = ProteinDataset(val_files)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # Initialize the Mystery-equivariant transformer model
    model = SE3Transformer(hidden_dim=128).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Initialize the diffusion process
    diffusion = ProteinDiffusion(device=device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create directory for saving samples
    os.makedirs("samples", exist_ok=True)

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter()

    step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch, mask in train_dataloader:
            # Move batch to the specified device (GPU/CPU)
            batch, mask = batch.to(device), mask.to(device)

            # Sample a random timestep for each item in the batch
            t = torch.randint(0, diffusion.num_steps, (batch.shape[0],), device=device)

            # Apply forward diffusion to get noisy samples and the added noise
            x_t, noise = diffusion.q_sample(batch, t)

            if step % visualization_interval == 0:
                visualize_batch(batch, x_t, step)

            # Predict the noise using the model
            predicted_noise = model(x_t, t, mask)

            # Calculate the loss (mean squared error between predicted and actual noise)
            loss = F.mse_loss(
                predicted_noise * mask.unsqueeze(-1), noise * mask.unsqueeze(-1)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss
            train_loss += loss.item()

            # Increment step counter
            step += 1

            # Print training progress
            print(f"train loss: {loss.item():.4f}, step: {step}")

            # Log to TensorBoard
            writer.add_scalar("Loss/train", loss.item(), step)

            if step % sample_interval == 0:
                generate_and_save_sample(model, diffusion, sample_length, step, device)

        # Calculate average training loss for the epoch
        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmsd = 0.0
        with torch.no_grad():
            for batch, mask in val_dataloader:
                batch, mask = batch.to(device), mask.to(device)
                t = torch.randint(
                    0, diffusion.num_steps, (batch.shape[0],), device=device
                )
                x_t, noise = diffusion.q_sample(batch, t)
                predicted_noise = model(x_t, t, mask)
                val_loss += F.mse_loss(
                    predicted_noise * mask.unsqueeze(-1), noise * mask.unsqueeze(-1)
                ).item()

        val_loss /= len(val_dataloader)
        val_rmsd /= len(val_dataloader)
        writer.add_scalar("epoch/train/loss", train_loss, step)
        writer.add_scalar("epoch/val/loss", val_loss, step)
        writer.add_scalar("epoch/val/rmsd", val_rmsd, step)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSD: {val_rmsd:.4f}"
        )

        # Sample and save protein structure
        if (epoch + 1) % sample_interval == 0:
            sampled_protein = sample_protein(model, diffusion, sample_length, device)
            np.save(
                f"samples/protein_epoch_{epoch+1}.npy",
                sampled_protein.cpu().detach().numpy(),
            )

        # Save model checkpoint
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
