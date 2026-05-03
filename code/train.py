"""
Training loop and evaluation.

Owner: Person 2
Deliverable: train PatchTST with Adam + MSE, early stopping, checkpoint saving;
             eval reports MSE/MAE on val and test.
"""

# High-level flow
# 1. Build train / val DataLoaders from dataset.py
# 2. Instantiate PatchTST from model.py
# 3. Create a Trainer, call trainer.fit()
# 4. Trainer saves the best checkpoint and returns a results dict

# The paper trains 1 model per (dataset, prediction_horizon) pair,

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import PatchTST
from dataset import TimeSeriesDataset  

# all hyper-parameters from paper Section 4.1 and Appendix A.1.4

DATASET_INFO = {
    # name          : (csv_relative_path,          n_features)
    "ETTh1"         : ("ETT-small/ETTh1.csv",       7),
    "ETTh2"         : ("ETT-small/ETTh2.csv",       7),
    "ETTm1"         : ("ETT-small/ETTm1.csv",       7),
    "ETTm2"         : ("ETT-small/ETTm2.csv",       7),
    "Weather"       : ("weather/weather.csv",       21),
    "Traffic"       : ("traffic/traffic.csv",      862),
    "Electricity"   : ("electricity/electricity.csv", 321),
    "ILI"           : ("illness/national_illness.csv",  7),
}

# Small datasets use reduced model size to avoid overfitting (Appendix A.1.4)
SMALL_DATASETS = {"ETTh1", "ETTh2", "ILI"}


def paper_cfg() -> Dict:
    return dict(
        #  Data 
        dataset         = "ETTh1",
        data_root       = "../../data/all_six_datasets",
        seq_len         = 336,       # look-back window L
        pred_len        = 96,        # forecast horizon T

        #  Patching 
        patch_len       = 16,        # patch length P
        stride          = 8,         # stride S (patches overlap by P-S steps)

        #  Transformer ─
        d_model         = 128,       # latent dimension D
        n_heads         = 16,        # attention heads H
        n_layers        = 3,         # number of encoder layers
        d_ff            = 256,       # feed-forward inner dimension F (=2*D)
        dropout         = 0.2,       # dropout probability

        #  Training 
        batch_size      = 128,
        lr              = 1e-4,
        epochs          = 100,
        patience        = 10,        # early-stopping patience (val MSE)
        #  misc 
        device          = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir  = "../results/checkpoints",
        log_dir         = "../results/logs",
        num_workers     = 2,
        seed            = 2021,
    )


# helpers

def mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return ((pred - true) ** 2).mean()


def mae(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (pred - true).abs().mean()


# data factory

def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:

    data_root = Path(cfg["data_root"])
    csv_path  = data_root / DATASET_INFO[cfg["dataset"]][0]

    def make_loader(split: str, shuffle: bool) -> DataLoader:
        ds = TimeSeriesDataset( 
            csv_path = str(csv_path),
            split    = split,
            seq_len  = cfg["seq_len"],
            pred_len = cfg["pred_len"],
        )
        return DataLoader(
            ds,
            batch_size  = cfg["batch_size"],
            shuffle     = shuffle,
            num_workers = cfg["num_workers"],
            pin_memory  = cfg["device"] == "cuda",
        )

    return make_loader("train", True), make_loader("val", False), make_loader("test", False)


# Model factory

def build_model(cfg: Dict) -> nn.Module:
    # For small datasets (ETTh1, ETTh2, ILI) we override d_model / n_heads / d_ff, to match Appendix A.1.4 (D=16, H=4, F=128).
    n_features = DATASET_INFO[cfg["dataset"]][1]

    # From paper: reduced capacity for small datasets to avoid overfitting
    d_model = cfg["d_model"]
    n_heads = cfg["n_heads"]
    d_ff    = cfg["d_ff"]
    if cfg["dataset"] in SMALL_DATASETS:
        d_model, n_heads, d_ff = 16, 4, 128

    model = PatchTST(
        seq_len    = cfg["seq_len"],
        pred_len   = cfg["pred_len"],
        patch_len  = cfg["patch_len"],
        stride     = cfg["stride"],
        n_features = n_features,
        d_model    = d_model,
        n_heads    = n_heads,
        n_layers   = cfg["n_layers"],
        d_ff       = d_ff,
        dropout    = cfg["dropout"],
    )
    return model.to(cfg["device"])


# Logger

class CSVLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists()

    def log(self, row: Dict):
        write_header = not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)



class Trainer:
    def __init__(self, cfg: Dict, model: Optional[nn.Module] = None):
        self.cfg = cfg
        self._set_seed(cfg["seed"])

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(cfg)
        self.model     = model if model is not None else build_model(cfg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.criterion = nn.MSELoss()
        self.device    = torch.device(cfg["device"])

        tag = f"{cfg['dataset']}_L{cfg['seq_len']}_T{cfg['pred_len']}"
        ckpt_dir = Path(cfg["checkpoint_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir  = Path(cfg["log_dir"]);        log_dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_path  = ckpt_dir / f"best_{tag}.pt"
        self.logger     = CSVLogger(str(log_dir / f"train_{tag}.csv"))


    @staticmethod
    def _set_seed(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        # Run one pass over `loader`.

        self.model.train(train)
        total_mse = total_mae = n_batches = 0.0

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for x, y in loader:
                # x : (B, M, seq_len)   — look-back window
                # y : (B, M, pred_len)  — forecast target
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                pred = self.model(x)       # (B, M, pred_len)

                loss = self.criterion(pred, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Gradient clipping (good practice for transformers)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                with torch.no_grad():
                    total_mse += mse(pred, y).item()
                    total_mae += mae(pred, y).item()
                n_batches += 1

        return total_mse / n_batches, total_mae / n_batches

    # training loop
    def fit(self) -> Dict:
        # Train for up to cfg['epochs'] epochs with early stopping on val MSE.

        cfg        = self.cfg
        best_val   = float("inf")
        best_epoch = 0
        patience   = cfg["patience"]
        wait       = 0
        t0         = time.time()

        print(
            f"Training PatchTST | dataset={cfg['dataset']} "
            f"L={cfg['seq_len']} T={cfg['pred_len']} | "
            f"device={cfg['device']}"
        )
        print(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(1, cfg["epochs"] + 1):
            t_ep = time.time()

            train_mse, train_mae = self._run_epoch(self.train_loader, train=True)
            val_mse,   val_mae   = self._run_epoch(self.val_loader,   train=False)

            ep_time = time.time() - t_ep

            #  Logging ─
            self.logger.log({
                "epoch"     : epoch,
                "train_mse" : round(train_mse, 6),
                "train_mae" : round(train_mae, 6),
                "val_mse"   : round(val_mse,   6),
                "val_mae"   : round(val_mae,   6),
                "epoch_s"   : round(ep_time,   2),
            })

            print(
                f"Epoch {epoch:03d}/{cfg['epochs']} | "
                f"train MSE={train_mse:.4f} MAE={train_mae:.4f} | "
                f"val   MSE={val_mse:.4f} MAE={val_mae:.4f} | "
                f"{ep_time:.1f}s"
            )

            #  Checkpoint 
            if val_mse < best_val:
                best_val   = val_mse
                best_epoch = epoch
                wait       = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f"  ===== New best val MSE={best_val:.4f} — checkpoint saved")
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stopping triggered (patience={patience})")
                    break

        total_time = time.time() - t0

        #  Final evaluation on test set ─
        print("\nLoading best checkpoint for test evaluation …")
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        test_mse, test_mae = self._run_epoch(self.test_loader, train=False)

        print("=" * 60)
        print(
            f"FINAL TEST  MSE={test_mse:.4f}  MAE={test_mae:.4f}  "
            f"(best epoch={best_epoch}, total={total_time/60:.1f} min)"
        )
        print("=" * 60)

        return dict(
            best_val_mse = best_val,
            best_val_mae = None,        # stored per epoch in CSV
            test_mse     = test_mse,
            test_mae     = test_mae,
            best_epoch   = best_epoch,
            total_time_s = total_time,
        )


    #  Convenience: load a checkpoint and evaluate ─
    def evaluate(self, split: str = "test") -> Tuple[float, float]:
        loader_map = {
            "train" : self.train_loader,
            "val"   : self.val_loader,
            "test"  : self.test_loader,
        }
        assert split in loader_map, f"split must be one of {list(loader_map)}"
        return self._run_epoch(loader_map[split], train=False)