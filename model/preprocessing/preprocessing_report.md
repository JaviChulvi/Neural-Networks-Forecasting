# Financial data preprocessing report

## Objective

This report applies the financial preprocessing ideas from the first workshop of block 3 to the forecasting dataset.

The original repository stores adjusted close prices and log returns. To build activity-based bars, daily OHLCV data is re-downloaded from Yahoo Finance for the same universe of assets.

## Important limitation

Yahoo Finance does not provide transaction-level trades in this dataset. Therefore, real tick bars cannot be constructed. Count bars are used as a daily proxy for tick bars, while volume bars and dollar bars are built using aggregated daily volume and dollar volume across the full asset universe.

## Generated bar types

- Time bars: original daily observations.
- Count bars: every fixed number of daily observations.
- Volume bars: days are grouped until a threshold of aggregated universe volume is reached.
- Dollar bars: days are grouped until a threshold of aggregated universe dollar volume is reached.

## Summary

| bar_type | description | threshold_type | threshold | n_bars | start_date | end_date | mean_days_per_bar | median_days_per_bar | mean_abs_return | return_std | return_skew | return_kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time | Daily time bars | calendar_day | 1 | 16196 | 1962-01-02 00:00:00 | 2026-05-08 00:00:00 | 1.45119 | 1 | 0.0116184 | 0.0171395 | -0.396927 | 19.3299 |
| count | Count bars: proxy for tick bars using daily observations | n_days | 16 | 1013 | 1962-01-23 00:00:00 | 2026-05-08 00:00:00 | 23.2026 | 23 | 0.0481764 | 0.0673803 | -0.69798 | 8.25252 |
| volume | Daily volume bars using aggregated universe volume | total_volume | 1.69751e+09 | 953 | 1962-07-10 00:00:00 | 2026-05-08 00:00:00 | 24.4884 | 18 | 0.0475594 | 0.0701738 | -0.391259 | 12.8745 |
| dollar | Daily dollar bars using aggregated universe dollar volume | total_dollar | 6.09484e+10 | 913 | 1981-01-05 00:00:00 | 2026-05-08 00:00:00 | 18.1568 | 10 | 0.0370422 | 0.062072 | 1.50115 | 23.9282 |

## Output files

Generated datasets are stored under `data/preprocessing/`.

The outputs include close prices, log returns, bar durations, summary statistics and plots for the different bar types.
