from __future__ import annotations
from torch.utils.data import DataLoader

def make_stream_loader(dataset, batch_size: int = 1, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
