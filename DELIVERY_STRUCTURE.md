# Delivery structure

The repository is organized around the two parts of the assignment.

## 1. Competition

Code:

- `model/mlp/`
- `model/cnn/`
- `model/rnn/`
- `model/mixtos/`
- `best_model_per_window.py`

Results:

- `reports/competition/`
- `best_model_per_window_report.md`
- `data/*` family result folders

## 2. Research and financial preprocessing

Code:

- `model/preprocessing/`

Results:

- `data/preprocessing/`
- `reports/preprocessing/`

Generated `.npz` sequence files are not included because they are large, but the scripts to recreate them are included.

## 3. Backtesting / portfolios for 2025

Code:

- `backtesting/scripts/`
- `backtesting/notebooks/`

Results:

- `data/backtest/`
- `reports/backtesting/`

## Notes

The original code paths are preserved to avoid breaking notebooks and scripts. The `reports/` folder is an additional review layer that gathers the final outputs required for the PDF/presentation.
