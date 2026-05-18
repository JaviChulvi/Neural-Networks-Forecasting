# Data and generated outputs

This folder contains the datasets and generated results used by the notebooks and scripts.

- `precios_close.parquet` and `returns.parquet`: base daily data.
- `mlp/`, `cnn/`, `rnn/`, `mixtos/`: model results and training histories.
- `preprocessing/`: activity-bar datasets and preprocessing summaries.
- `backtest/`: portfolio values, metrics and cumulative return plots.

Large local artifacts such as model weights (`*.keras`) and generated `.npz` preprocessing sequences should not be committed.
