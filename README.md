# CS 4782 Final Project — PatchTST Reproduction

Re-implementation of:
> **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers**
> Nie, Nguyen, Sinthong, Kalagnanam (ICLR 2023)
> [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

This repo is a from-scratch PyTorch re-implementation of PatchTST, plus five architectural extensions we built on top. Everything was done as part of CS 4782 at Cornell.

---

## Chosen Result

We targeted **Table 3, ETTh1, T=96** — look back 336 steps, predict 96 ahead. The paper hits **MSE = 0.375, MAE = 0.399**, about 15% better than the prior best (FEDformer). It's the headline result, so matching it was the most direct way to validate our implementation.

---

## Repository Contents

```
code/
  dataset.py      ETT data loading + PyTorch Dataset
  model.py        PatchTST model
  train.py        Training loop + evaluation
  utils.py        Shared helpers
  notebooks/      Colab notebooks (drivers for experiments)
data/             ETT datasets
results/          Plots, tables, logs
poster/           Poster PDF
report/           2-page report PDF
```

---

## Re-implementation Details

We built the model from scratch following Section 3.1 and Appendix A.1.4 of the paper. The key pieces: RevIN normalization per channel, patching the 336-step input into 42 overlapping patches of length 16 (stride 8), a linear projection into embedding space, and a 3-layer Transformer encoder. One non-obvious detail — the paper uses BatchNorm, not LayerNorm, which it only mentions in a footnote. It matters more than you'd expect.

**Dataset.** ETTh1 (Electricity Transformer Temperature, hourly) — 7 variables recorded from a Chinese electricity grid over ~17 months, 17,420 timesteps. Standard 70/10/20 split. We report MSE and MAE on the test set.

**Training.** Adam with OneCycleLR (max lr 1e-4), MSE loss, batch size 128, early stopping at patience 10. ETTh1 uses a reduced model (D=16, 4 heads) per Appendix A.1.4 to avoid overfitting.

**Extensions.** We added five extensions, each independently toggleable:

| ID | What it does | Why |
|----|-------------|-----|
| E1 | RoPE — rotate Q,K before attention, zero extra params | Relative position signal |
| E2 | Learnable per-channel scale/shift in RevIN | Preserve amplitude differences |
| E3 | Trend/residual split via moving average (kernel 25) | ETT has strong seasonal cycles |
| E4 | Cross-channel attention before the CI reshape | Oil temp and load readings are correlated |
| E5 | Patch importance weighting via MLP + softmax | Some time windows matter more |

The two trickiest parts: `BatchNorm1d` expects `(B,C,L)` but our tokens are `(B,N,D)`, so every BN call needed transposes. RoPE needed a full custom attention module since PyTorch's `nn.MultiheadAttention` doesn't let you intercept Q and K before the dot product.

---

## Reproduction Steps
**Notebook** All model training and evaluation occurs in the `code/notebooks/03_training.ipynb` python notebook. Each extension can be activated or deactivated by modifying the `cfg` configuration dictionary in the third cell. Running the final cell performs a full sweep across the specified T time steps on the ETTh1 dataset. Checkpoints go to `results/checkpoints/` and logs to `results/logs/`.

**Compute.** We trained on Google Colab GPUs (T4/A100 depending on availability). ETTh1 runs are fast enough to finish comfortably within a Colab session.

---

## Results / Insights

Our baseline (no extensions) closely reproduces the paper's reported MSE = 0.375, MAE = 0.399, and adding just RoPE pushes performance slightly further, marginally outperforming the original result.

Stacking all five extensions together was worse than RoPE alone. ETTh1 is a small dataset and adding that many parameters at once just causes overfitting. Decomposition (E3) also didn't help — a fixed moving-average window probably isn't the right tool for this dataset's noise structure.

Training curves and the full extension breakdown across T ∈ {96, 192, 336, 720} are in `results/`.

---

## Conclusion

PatchTST's two core ideas — patches and channel-independence — are simple and work really well. The reproduction was fairly smooth, though the BatchNorm footnote almost caught us off guard. RoPE turned out to be the best extension we tried, adding a meaningful boost with no extra parameters. If we were to continue, we'd test partial combinations of extensions rather than all five at once.

---

## References

- Nie et al. 2023. *A Time Series is Worth 64 Words.* ICLR. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Kim et al. 2022. *Reversible Instance Normalization for Accurate Time-Series Forecasting.* ICLR.
- Su et al. 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Zeng et al. 2023. *Are Transformers Effective for Time Series Forecasting?* AAAI.
- Official PatchTST code: https://github.com/yuqinie98/PatchTST
- ETT dataset: https://github.com/zhouhaoyi/ETDataset

---

## Acknowledgements

Course project for Cornell CS 4782 (Spring 2026), by Vinay Ivaturi, Aneesh Naresh, Arnav Kaul, and Jay Talwar.
