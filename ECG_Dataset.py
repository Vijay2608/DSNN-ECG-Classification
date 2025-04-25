import torch
from torch.utils.data import Dataset
import numpy as np

class ECGTransform:
    def __call__(self, x):
        x = x.astype(np.float32)
        if np.random.rand() > 0.5:
            x *= np.random.uniform(0.9, 1.1)
        if np.random.rand() > 0.7:
            t = np.linspace(0, 1, x.shape[-1])
            wander = 0.1 * np.sin(2 * np.pi * t * np.random.uniform(0.5, 2))
            x += wander
        if np.random.rand() > 0.6:
            x += np.random.normal(0, 0.05, x.shape)
        return x

class ECGDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y