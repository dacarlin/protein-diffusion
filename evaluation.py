import numpy as np

def compute_rmsd(pred_coords, true_coords):
    assert pred_coords.shape == true_coords.shape, "Shape mismatch"
    diff = pred_coords - true_coords
    return np.sqrt((diff ** 2).sum(axis=1).mean())
