# Financial data preprocessing

This folder applies preprocessing ideas from the financial data preprocessing workshop to the forecasting dataset.

The original forecasting repository stores adjusted close prices and log returns. Since the original source is Yahoo Finance, the notebooks re-download daily OHLCV data for the same 23 assets and build daily activity-based bars.

## Files

- `01_yahoo_ohlcv_audit.ipynb`: audits the existing dataset and downloads Yahoo OHLCV data.
- `02_activity_bars_yahoo.ipynb`: builds time bars, count bars, volume bars and dollar bars.
- `preprocessing_utils.py`: shared helper functions.
- `run_preprocessing.sh`: optional runner for the two notebooks.
- `preprocessing_report.md`: generated after running the second notebook.

## Output data

Generated files are written under:

- `data/preprocessing/`

The activity bars are adapted to the available daily Yahoo data. Real transaction-level tick bars cannot be built from Yahoo Finance data because transaction-level trades are not available.
