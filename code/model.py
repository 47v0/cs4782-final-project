"""
PatchTST model.

Components: instance norm + patching, projection + position embedding,
Transformer encoder, flatten + linear head.

Reference: Nie et al. 2023 (ICLR), "A Time Series is Worth 64 Words".
Forward pipeline matches Section 3.1 and Appendix A.1.5; hyperparameter
defaults follow Appendix A.1.4 (small-dataset override is set in train.py).
"""

import torch
import torch.nn as nn

class _ChannelAttention(nn.Module):
    """
    Lightweight channel attention: lets M variates interact before temporal modeling.

    Each time step t is treated as a batch item; the M channels are tokens.
    A learnable linear projection lifts each scalar value to d_model, runs
    multi-head self-attention, then projects back to a scalar correction.
    LayerNorm before attention (pre-norm) stabilises training.
    """

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.proj_in  = nn.Linear(1, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.attn     = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.proj_out = nn.Linear(d_model, 1)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, M, L)
        B, M, L = x.shape

        # treat channels as tokens
        x = x.permute(0, 2, 1)        # (B, L, M)
        x = x.reshape(B * L, M, 1)    # (B*L, M, 1)

        x = self.proj_in(x)           # (B*L, M, D)
        xn = self.norm(x)
        a, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + self.drop(a)

        x = self.proj_out(x)          # (B*L, M, 1)

        x = x.reshape(B, L, M)
        x = x.permute(0, 2, 1)        # (B, M, L)

        return x

"""
Extension: Rotary Position Embedding (RoPE) -- Su et al. 2021, arXiv:2104.09864.
  Enabled via use_rope=True in the constructor.  When active, the learnable
  additive pos_emb is dropped and position is encoded by rotating Q and K
  before every attention dot-product, so the model sees *relative* patch
  distances rather than absolute indices.  Zero extra parameters.
"""

def _build_rope_cache(seq_len, head_dim, device, base=10_000.0):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )                                                         # (head_dim // 2,)
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)                  # (seq_len, head_dim // 2)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    """Rotation: [x1, x2] -> [x1*cos - x2*sin,  x1*sin + x2*cos]"""
    d  = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    cos = cos.unsqueeze(0).unsqueeze(0)   # (1, 1, N, head_dim//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


class _RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with RoPE applied to Q and K.
    Cache is built lazily and rebuilt whenever seq_len or device changes."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.q_proj    = nn.Linear(d_model, d_model, bias=False)
        self.k_proj    = nn.Linear(d_model, d_model, bias=False)
        self.v_proj    = nn.Linear(d_model, d_model, bias=False)
        self.o_proj    = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self._rope_cache = {}

    def _get_rope(self, N, device):
        key = (N, str(device))
        if key not in self._rope_cache:
            self._rope_cache[key] = _build_rope_cache(N, self.head_dim, device)
        return self._rope_cache[key]

    def forward(self, x):
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.head_dim

        def _split(proj):
            return proj(x).reshape(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)

        Q, K, V = _split(self.q_proj), _split(self.k_proj), _split(self.v_proj)
        cos, sin = self._get_rope(N, x.device)
        Q = _apply_rope(Q, cos, sin)
        K = _apply_rope(K, cos, sin)

        attn = (Q @ K.transpose(-2, -1)) * self.scale        # (B, H, N, N)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out  = (attn @ V).transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        return self.o_proj(out)



class _SeriesDecomp(nn.Module):
    """Centred moving-average decomposition.  Returns (trend, residual).

    AvgPool1d with symmetric padding preserves sequence length; an even kernel
    can add one extra sample, which is trimmed.
    """

    def __init__(self, kernel_size=25):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2)

    def forward(self, x):
        # x: (B*M, L) -- already channel-flattened
        trend = self.avg(x.unsqueeze(1)).squeeze(1)   # pool over time
        trend = trend[:, : x.shape[-1]]               # trim if kernel is even
        return trend, x - trend



class _BNTransformerEncoderLayer(nn.Module):
    """
    Post-norm transformer encoder layer with BatchNorm1d in place of LayerNorm.

    Per the PatchTST paper footnote (p.5, citing Zerveas et al. 2021), BatchNorm
    outperforms LayerNorm in time-series transformer encoders. PyTorch's
    nn.TransformerEncoderLayer is hardcoded to LayerNorm, so we hand-roll the
    BN variant here.

    BN1d expects (B, C, L); our token tensor is (B, N, D). We transpose to
    (B, D, N) for the BN call and back.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, use_rope=True):
        super().__init__()
        if use_rope:
            self.attn = _RoPEMultiheadAttention(d_model, nhead, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                d_model, nhead, dropout=dropout, batch_first=True
            )
        self.use_rope = use_rope

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.bn1  = nn.BatchNorm1d(d_model)
        self.bn2  = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                      # x: (B, N, D)
        if self.use_rope:
            a = self.attn(x)
        else:
            a, _ = self.attn(x, x, x, need_weights=False)
        x = self.bn1((x + self.drop(a)).transpose(1, 2)).transpose(1, 2)
        f = self.ff(x)
        x = self.bn2((x + self.drop(f)).transpose(1, 2)).transpose(1, 2)
        return x


# ── Main model ────────────────────────────────────────────────────────────────

class PatchTST(nn.Module):
    """
    Channel-independent patch transformer for multivariate forecasting.

    Forward pass, given x of shape (B, M, L):
      1. RevIN: normalise per-instance per-channel; optionally apply learnable
         affine γ·x̂ + β (E2).  Save (mean, std) for the inverse pass.
      2. Channel-independence reshape: (B, M, L) -> (B*M, L).
      3. [if use_decomp] split (B*M, L) into trend and residual (E3);
         steps 4-6 run once per stream (shared weights) and outputs are summed.
      4. Patching: replicate-pad by `stride`, unfold to (B*M, N, P).
      5. Linear projection P -> D; add pos_emb if not use_rope (E1).
      6. Transformer encoder with BN; attention uses RoPE if use_rope=True.
      7. Flatten + Linear head: (B*M, N, D) -> (B*M, T).
      8. Reshape to (B, M, T) and undo RevIN affine + normalisation.

    N = (L - P) // S + 2.
    """

    def __init__(self,
                 seq_len, pred_len, patch_len, stride,
                 n_features, d_model, n_heads, n_layers,
                 d_ff, dropout, norm_type="batch",
                 use_rope=True,
                 use_revin_affine=True,   
                 use_decomp=True,       
                 use_channel_attn=True,
                 use_adap_patch=True,
                 decomp_kernel=25):       
        super().__init__()
        if norm_type not in {"layer", "batch"}:
            raise ValueError(f"norm_type must be 'layer' or 'batch', got {norm_type!r}")

        self.seq_len          = seq_len
        self.pred_len         = pred_len
        self.patch_len        = patch_len
        self.stride           = stride
        self.n_features       = n_features
        self.d_model          = d_model
        self.norm_type        = norm_type
        self.use_rope         = use_rope
        self.use_revin_affine = use_revin_affine
        self.use_decomp       = use_decomp
        self.use_channel_attn = use_channel_attn
        self.use_adap_patch   = use_adap_patch
        self.n_patches        = (seq_len - patch_len) // stride + 2
        self._revin_eps       = 1e-5

        # E2: one scale and one shift per channel, shared across B and T.
        if use_revin_affine:
            self.gamma = nn.Parameter(torch.ones (1, n_features, 1))
            self.beta  = nn.Parameter(torch.zeros(1, n_features, 1))

        # E3: decomposition module (no learnable parameters).
        if use_decomp:
            self.decomp = _SeriesDecomp(decomp_kernel)

        self.patch_proj = nn.Linear(patch_len, d_model)

        # pos_emb only exists when RoPE is off; RoPE carries position in attn.
        if not use_rope:
            self.pos_emb = nn.Parameter(
                torch.randn(1, self.n_patches, d_model) * 0.02
            )

        if norm_type == "batch":
            self.encoder = nn.Sequential(*[
                _BNTransformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                           use_rope=use_rope)
                for _ in range(n_layers)
            ])
        else:
            # LayerNorm path kept for backward compat; use_rope not wired here.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model         = d_model,
                nhead           = n_heads,
                dim_feedforward = d_ff,
                dropout         = dropout,
                activation      = "gelu",
                batch_first     = True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # E4: channel attention
        if use_channel_attn:
            self.channel_attn = _ChannelAttention(
              d_model=d_model,
              n_heads=n_heads,
              dropout=dropout,
            )

        # E5: adaptive patching
        if self.use_adap_patch:
          self.patch_importance = nn.Sequential(
              nn.Linear(self.patch_len, self.d_model),
              nn.GELU(),
              nn.Linear(self.d_model, 1)
          )
          
        self.flatten = nn.Flatten(start_dim=-2)
        self.head    = nn.Linear(self.n_patches * d_model, pred_len)

    def _encode(self, x):
        last = x[:, -1:].expand(-1, self.stride)
        x    = torch.cat([x, last], dim=-1)
        x    = x.unfold(-1, self.patch_len, self.stride) # (B*M, N, P)

        if self.use_adap_patch:
          weights = self.patch_importance(x)         # (B*M, N, 1)
          weights = torch.softmax(weights, dim=1)    # normalize across patches
          x = x * weights                           # reweight patches

        x    = self.patch_proj(x)                            # (B*M, N, D)
        if not self.use_rope:
            x = x + self.pos_emb
        x = self.encoder(x)                                  # (B*M, N, D)
        x = self.flatten(x)                                  # (B*M, N*D)
        return self.head(x)                                  # (B*M, T)

    def forward(self, x):
        # x: (B, M, L)
        B, M, L = x.shape

        # 1. RevIN normalisation.
        mu  = x.mean(dim=-1, keepdim=True)                   # (B, M, 1)
        sig = x.std (dim=-1, keepdim=True) + self._revin_eps
        x   = (x - mu) / sig
        if self.use_revin_affine:                             # E2: affine in
            x = self.gamma * x + self.beta

        # E4: channel attention
        if self.use_channel_attn:
          x = x + self.channel_attn(x)

        # 2. Channel-independence: merge M into batch.
        x = x.reshape(B * M, L)                              # (B*M, L)

        # 3-7. Encode; if decomposing (E3), run each stream and sum.
        if self.use_decomp:
            trend, residual = self.decomp(x)
            out = self._encode(trend) + self._encode(residual)
        else:
            out = self._encode(x)                            # (B*M, T)

        # 8. Reshape + undo RevIN.
        x = out.reshape(B, M, self.pred_len)                 # (B, M, T)
        if self.use_revin_affine:                             # E2: affine out
            x = (x - self.beta) / (self.gamma + self._revin_eps)
        x = x * sig + mu

        return x