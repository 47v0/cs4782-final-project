# Datasets

The 8 multivariate time-series datasets used in the PatchTST paper:

| Dataset | Features | Frequency | Notes |
|---|---|---|---|
| ETTh1, ETTh2 | 7 | hourly | Electricity transformer temperature |
| ETTm1, ETTm2 | 7 | 15 min | Electricity transformer temperature |
| Weather | 21 | 10 min | German meteorological indicators |
| Traffic | 862 | hourly | San Francisco freeway road occupancy |
| Electricity | 321 | hourly | Customer hourly consumption |
| Illness (ILI) | 7 | weekly | US patient counts with influenza-like illness |

## Source

All 6 dataset folders are bundled in `all_six_datasets.zip` (committed in this directory for convenience). Same files are also available from:

- **ETT** (4 CSVs): https://github.com/zhouhaoyi/ETDataset (folder `ETT-small/`)
- **Weather, Traffic, Electricity, Illness, Exchange-rate**: https://github.com/yuqinie98/PatchTST (the official repo distributes these together)

## Local setup

To unpack the zip in place:

```bash
unzip data/all_six_datasets.zip -d data/
```

This produces:

```
data/all_six_datasets/
  ETT-small/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  weather/weather.csv
  traffic/traffic.csv
  electricity/electricity.csv
  illness/national_illness.csv
  exchange_rate/exchange_rate.csv
```

The unpacked folder is gitignored (see `.gitignore`).

## Colab setup

When running on Colab, clone the repo into Drive then unzip into a working directory:

```python
!unzip -q /content/drive/MyDrive/cs4782-final-project/data/all_six_datasets.zip \
       -d /content/data
```
