# my_dataset.py
import zarr
import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self):
        self.zarr_path = '/data1/lesliec/sarthak/data/gpn/99.zarr'

    def __getitem__(self, idx):
        z_store = zarr.open(self.zarr_path, mode='r')
        print(z_store['1'][:10])
        return idx

    def __len__(self):
        return 1000
