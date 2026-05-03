"""
PatchTST model.

Owner: Person 1
Deliverable: an nn.Module taking (B, 7, 336) and returning (B, 7, 96).
Components: instance norm + patching, projection + position embedding,
Transformer encoder, flatten + linear head.
"""

import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(self,
                 seq_len, pred_len, patch_len, stride,
                 n_features, d_model, n_heads, n_layers,
                 d_ff, dropout):
        super().__init__()

        self.pred_len = pred_len
        self.n_features = n_features

        # super simple "fake transformer"
        self.backbone = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x: (B, M, L)
        B, M, L = x.shape

        x = x.view(B * M, L)              # flatten channels
        out = self.backbone(x)            # (B*M, pred_len)
        out = out.view(B, M, self.pred_len)

        return out