"""
Custom dataset to load multiple datasets at once
Taken from https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649
"""

import torch.utils.data as data


class ConcatDataset(data.Dataset):
    """Custom dataset to load multiple datasets at once"""
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
