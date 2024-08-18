import numpy as np
import torch

def compute_rmsd(pred_coords, true_coords):
    assert pred_coords.shape == true_coords.shape, "Shape mismatch"
    diff = pred_coords - true_coords
    return np.sqrt((diff ** 2).sum(axis=1).mean())

def evaluate_model(model, dataloader):
    model.eval()
    rmsds = []
    with torch.no_grad():
        for batch in dataloader:
            noisy_batch = batch + torch.randn_like(batch) * 0.1
            output = model(noisy_batch)
            batch_rmsd = compute_rmsd(output.cpu().numpy(), batch.cpu().numpy())
            rmsds.append(batch_rmsd)
    avg_rmsd = np.mean(rmsds)
    print(f"Average RMSD: {avg_rmsd:.4f}")
    return avg_rmsd
