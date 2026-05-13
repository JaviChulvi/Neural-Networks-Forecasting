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

| bar_type   | description                                               | threshold_type   |   threshold |   n_bars | start_date          | end_date            |   mean_days_per_bar |   median_days_per_bar |   mean_abs_return |   return_std |   return_skew |   return_kurtosis |
|:-----------|:----------------------------------------------------------|:-----------------|------------:|---------:|:--------------------|:--------------------|--------------------:|----------------------:|------------------:|-------------:|--------------:|------------------:|
| time       | Daily time bars                                           | calendar_day     | 1           |    16196 | 1962-01-02 00:00:00 | 2026-05-08 00:00:00 |             1.45119 |                     1 |         0.0116184 |    0.0171395 |     -0.396927 |          19.3299  |
| count      | Count bars: proxy for tick bars using daily observations  | n_days           | 4           |     4049 | 1962-01-05 00:00:00 | 2026-05-08 00:00:00 |             5.80509 |                     6 |         0.0242089 |    0.0345307 |     -0.32274  |           8.04026 |
| volume     | Daily volume bars using aggregated universe volume        | total_volume     | 4.24378e+08 |     3369 | 1962-03-01 00:00:00 | 2026-05-08 00:00:00 |             6.96081 |                     5 |         0.0251991 |    0.0372088 |     -0.299662 |          11.9418  |
| dollar     | Daily dollar bars using aggregated universe dollar volume | total_dollar     | 1.52528e+10 |     2986 | 1972-11-03 00:00:00 | 2026-05-08 00:00:00 |             6.5474  |                     3 |         0.0205583 |    0.0355322 |      1.87298  |          65.805   |

## Output files

Generated datasets are stored under `data/preprocessing/`.

The outputs include close prices, log returns, bar durations, summary statistics and plots for the different bar types.

<!-- PREPROCESSED_SEQUENCES_SECTION_START -->

## Forecasting sequence generation

To connect the financial preprocessing stage with the forecasting task, the transformed bar datasets were also converted into supervised learning sequences.

For each bar representation:

- time bars
- count bars
- volume bars
- dollar bars

we generated input/output windows using the same combinations as in the main competition:

- input windows: 5, 10, 30 and 90 bars
- output windows: 1, 5, 30 and 90 bars

For each sample, `X` contains the past `input_window` returns, and `y` contains the average return over the next `output_window` bars for the 23 assets.

The resulting files are stored under:

- `data/preprocessing/sequences/time/`
- `data/preprocessing/sequences/count/`
- `data/preprocessing/sequences/volume/`
- `data/preprocessing/sequences/dollar/`

A global summary is available in:

- `data/preprocessing/preprocessed_sequences_summary.csv`
- `data/preprocessing/preprocessed_sequences_sample_matrix.csv`

## Impact on the forecasting dataset

The different bar construction methods produce datasets with different numbers of observations.

Time bars keep the original daily frequency and therefore provide the largest number of samples. Count, volume and dollar bars aggregate several calendar days into one activity-based bar, reducing the number of observations but making each observation represent a more comparable amount of market activity.

This affects the forecasting problem in two ways:

1. The number of training samples is reduced for activity-based bars.
2. The meaning of the input and output windows changes: a 30-bar window no longer necessarily means 30 calendar days, but 30 activity-based bars.

Therefore, the transformed datasets are directly usable by the same forecasting logic as the competition, but the interpretation of the temporal windows must be adapted.

## Limitations

The original repository only stored adjusted close prices and log returns. To apply the financial preprocessing techniques, daily OHLCV data was re-downloaded from Yahoo Finance for the same assets.

Yahoo Finance does not provide transaction-level data in this setup. Therefore, real tick bars cannot be constructed. Count bars are used as a daily proxy for tick bars, while volume bars and dollar bars are computed from aggregated daily volume and dollar volume across the full asset universe.

The activity bars are built on a common multi-asset calendar. This keeps the resulting matrices compatible with the multivariate forecasting setup, but it is an approximation: each asset does not receive its own independent activity clock.

<!-- PREPROCESSED_SEQUENCES_SECTION_END -->
