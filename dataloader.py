import os
import torch
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser

class ProteinDataset(Dataset):
    def __init__(self, pdb_dir):
        self.pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir)]
        self.parser = PDBParser(QUIET=True)

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        structure = self.parser.get_structure('protein', self.pdb_files[idx])
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
        coords = torch.tensor(coords, dtype=torch.float32)
        return coords

def get_dataloader(pdb_dir, batch_size=2):
    dataset = ProteinDataset(pdb_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
