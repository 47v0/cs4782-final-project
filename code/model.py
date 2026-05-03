"""
PatchTST model.

Owner: Person 1
Deliverable: an nn.Module taking (B, 7, 336) and returning (B, 7, 96).
Components: instance norm + patching, projection + position embedding,
Transformer encoder, flatten + linear head.

Reference: Nie et al. 2023 (ICLR), "A Time Series is Worth 64 Words".
Forward pipeline matches Section 3.1 and Appendix A.1.5; hyperparameter
defaults follow Appendix A.1.4 (small-dataset override is set in train.py).
"""

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    Channel-independent patch transformer for multivariate forecasting.

    Forward pass, given x of shape (B, M, L):
      1. RevIN: per-instance per-channel normalize across time, save (mean, std).
      2. Channel-independence reshape: (B, M, L) -> (B*M, L).
      3. Patching: replicate-pad by `stride`, unfold to (B*M, N, P).
      4. Linear projection P -> D and learnable additive position embedding.
      5. Vanilla transformer encoder (LayerNorm, GELU, batch-first).
      6. Flatten + Linear head: (B*M, N, D) -> (B*M, T).
      7. Reshape to (B, M, T) and undo RevIN with the saved (mean, std).

    Number of patches N = (L - P) // S + 2, where the +2 accounts for the
    standard sliding-window count plus one extra patch from the replicate-pad.
    """

    def __init__(self,
                 seq_len, pred_len, patch_len, stride,
                 n_features, d_model, n_heads, n_layers,
                 d_ff, dropout):
        super().__init__()

        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_features = n_features
        self.d_model    = d_model
        self.n_patches  = (seq_len - patch_len) // stride + 2

        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb    = nn.Parameter(
            torch.randn(1, self.n_patches, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.head    = nn.Linear(self.n_patches * d_model, pred_len)

        self._revin_eps = 1e-5

    def forward(self, x):
        # x: (B, M, L)
        B, M, L = x.shape

        # 1. RevIN at input.
        mu  = x.mean(dim=-1, keepdim=True)                            # (B, M, 1)
        sig = x.std (dim=-1, keepdim=True) + self._revin_eps          # (B, M, 1)
        x   = (x - mu) / sig

        # 2. Channel-independence: merge channel into batch.
        x = x.reshape(B * M, L)

        # 3. Patching: replicate the last value `stride` times, then unfold.
        last = x[:, -1:].expand(-1, self.stride)
        x    = torch.cat([x, last], dim=-1)                           # (B*M, L+S)
        x    = x.unfold(-1, self.patch_len, self.stride)              # (B*M, N, P)

        # 4. Project + add position embedding.
        x = self.patch_proj(x)                                        # (B*M, N, D)
        x = x + self.pos_emb

        # 5. Transformer encoder.
        x = self.encoder(x)                                           # (B*M, N, D)

        # 6. Flatten + head.
        x = self.flatten(x)                                           # (B*M, N*D)
        x = self.head(x)                                              # (B*M, T)

        # 7. Un-merge channel + undo RevIN.
        x = x.reshape(B, M, self.pred_len)                            # (B, M, T)
        x = x * sig + mu                                              # broadcast across T

        return x
