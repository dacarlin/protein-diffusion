import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
import numpy as np


class ProteinDataset(Dataset):
    """Dataset that lets us immediately begin stepping and holds nothing in memory
    (instead reading from disk every time)"""

    def __init__(self, pdb_files, max_length=512):
        self.pdb_files = pdb_files
        self.max_length = max_length
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        structure = self.parser.get_structure("protein", self.pdb_files[idx])
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        coords.append(residue["CA"].get_coord())
        coords = torch.tensor(np.array(coords[:self.max_length]), dtype=torch.float32)
        return coords


class InMemoryProteinDataset(Dataset):
    """Dataset that holds everything in memory, and that allows graceful handling
    of bad input files"""

    def __init__(self, pdb_files, max_length=512):
        self.pdb_files = pdb_files
        self.max_length = max_length
        self.parser = PDBParser(QUIET=True)
        self.samples = []
        for idx in range(len(self.pdb_files)):
            structure = self.parser.get_structure("protein", self.pdb_files[idx])
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if "CA" in residue:
                            coords.append(residue["CA"].get_coord())
            coords = torch.tensor(np.array(coords[:self.max_length]), dtype=torch.float32)
            self.samples.append(coords)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate_fn(batch):
    """Pad input tensors with zeros to the maximum length of the biggest input"""
    max_len = max(coords.shape[0] for coords in batch)
    padded_batch = []
    for coords in batch:
        pad_size = max_len - coords.shape[0]
        padded_coords = torch.nn.functional.pad(
            coords, (0, 0, 0, pad_size), mode="constant", value=0
        )
        padded_batch.append(padded_coords)
    return torch.stack(padded_batch, dim=0)
