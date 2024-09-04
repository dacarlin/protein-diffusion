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
        coords = torch.tensor(np.array(coords[: self.max_length]), dtype=torch.float32)
        mask = torch.ones(len(coords), dtype=bool)

        # Pad if necessary
        if len(coords) < self.max_length:
            pad_size = self.max_length - len(coords)
            coords = np.pad(
                coords, ((0, pad_size), (0, 0)), mode="constant", constant_values=0
            )
            mask = np.pad(mask, (0, pad_size), mode="constant", constant_values=False)
            coords = torch.tensor(coords, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.bool)

        return coords, mask


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
            coords = torch.tensor(
                np.array(coords[: self.max_length]), dtype=torch.float32
            )
            self.samples.append(coords)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def pad_collate_fn(batch):
    # No need for additional padding as it's done in __getitem__
    coords, masks = zip(*batch)
    return torch.stack(coords, dim=0), torch.stack(masks, dim=0)
