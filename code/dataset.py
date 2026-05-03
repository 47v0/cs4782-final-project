"""
ETT dataset loading and PyTorch Dataset/DataLoader.

Owner: Person 1
Deliverable: a DataLoader yielding (look_back, forecast) batches with
             shapes (B, 7, 336) and (B, 7, 96).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Paper splits for ETTh1/ETTh2 — 12/4/4 months at hourly resolution.
# Convention from the Informer paper, also hardcoded in the official PatchTST code.
_TRAIN_END = 12 * 30 * 24  # 8640
_VAL_END = _TRAIN_END + 4 * 30 * 24  # 11520
_TEST_END = _VAL_END + 4 * 30 * 24  # 14400

# Numeric columns of ETTh1 (the `date` column is dropped).
_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset over a chronological split of ETT-style hourly data.

    Each item is (x, y) where:
        x : tensor of shape (n_features, seq_len)  — past `seq_len` hours
        y : tensor of shape (n_features, pred_len) — next `pred_len` hours
    x and y are contiguous in time: y starts exactly at the row after x ends.

    Splits use the standard PatchTST convention (per the official code):
        train: rows [0,                       TRAIN_END)
        val:   rows [TRAIN_END  - seq_len,    VAL_END)
        test:  rows [VAL_END    - seq_len,    TEST_END)
    The -seq_len offset on val/test means the FIRST window's prediction lands
    exactly at the split boundary (the actual first val/test row).

    Per-channel z-score is fit on the train rows only and applied uniformly.
    """

    def __init__(self, csv_path, split, seq_len, pred_len, features=None):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features if features is not None else _FEATURES

        df = pd.read_csv(csv_path)
        full = df[self.features].to_numpy(dtype=np.float32)

        # Fit z-score on TRAIN rows only, apply to the entire series.
        train_rows = full[:_TRAIN_END]
        mean = train_rows.mean(axis=0, keepdims=True)
        std = train_rows.std(axis=0, keepdims=True) + 1e-8
        full = (full - mean) / std

        if split == "train":
            data = full[:_TRAIN_END]
        elif split == "val":
            data = full[_TRAIN_END - seq_len : _VAL_END]
        else:
            data = full[_VAL_END - seq_len : _TEST_END]

        self.data = torch.from_numpy(data).float()  # (rows, n_features)
        self.mean = torch.from_numpy(mean).float()  # (1, n_features)
        self.std = torch.from_numpy(std).float()  # (1, n_features)

    def __len__(self):
        return max(0, self.data.shape[0] - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len

        # Slice (time, features) then transpose → (features, time).
        x = self.data[s_begin:s_end].transpose(0, 1).contiguous()
        y = self.data[s_end:r_end].transpose(0, 1).contiguous()
        return x, y
