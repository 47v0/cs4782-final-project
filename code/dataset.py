"""
ETT dataset loading and PyTorch Dataset/DataLoader.

Owner: Person 1
Deliverable: a DataLoader yielding (look_back, forecast) batches with
             shapes (B, 7, 336) and (B, 7, 96).
"""

import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, split, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.M = 7  # match ETTh1 feature count
        self.N = 1000  # number of samples

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = torch.randn(self.M, self.seq_len)
        y = torch.randn(self.M, self.pred_len)
        return x, y
