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


torch.manual_seed(99)
seed(99)


def main():
    # run params
    pdb_dir = "data/dompdb"
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    sample_length = 100  # length of protein to sample
    sample_interval = 10
    max_samples = 45_000
    fit_one_sample = False

    # create device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else 
        "cuda" if torch.cuda.is_available() else 
        "cpu")
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
    val_dataset = ProteinDataset(train_files)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    # model = DenoisingDiffusionModel().to(device)
    model = SE3Transformer(hidden_dim=128).to(device)
    # model = torch.compile(model)
    diffusion = ProteinDiffusion(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create directory for saving samples
    os.makedirs("samples", exist_ok=True)
    writer = SummaryWriter()

    step = 0
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
            step += 1
            print(f"train loss: {loss.item():.4f}, step: {step}")

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmsd = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                t = torch.randint(
                    0, diffusion.num_steps, (batch.shape[0],), device=device
                )
                x_t, noise = diffusion.q_sample(batch, t)
                predicted_noise = model(x_t, t)
                val_loss += F.mse_loss(predicted_noise, noise).item()
                val_rmsd += compute_rmsd(batch.cpu().numpy(), x_t.cpu().numpy())

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
