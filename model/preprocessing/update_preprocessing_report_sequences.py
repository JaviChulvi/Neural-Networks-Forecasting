from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
report_path = PROJECT_ROOT / "model" / "preprocessing" / "preprocessing_report.md"

start_marker = "<!-- PREPROCESSED_SEQUENCES_SECTION_START -->"
end_marker = "<!-- PREPROCESSED_SEQUENCES_SECTION_END -->"

section = f"""
{start_marker}

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

{end_marker}
""".strip() + "\n"

if report_path.exists():
    content = report_path.read_text()
else:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = "# Financial preprocessing report\n\n"

if start_marker in content and end_marker in content:
    before = content.split(start_marker)[0].rstrip()
    after = content.split(end_marker, 1)[1].lstrip()
    new_content = before + "\n\n" + section + "\n" + after
else:
    new_content = content.rstrip() + "\n\n" + section

report_path.write_text(new_content)
print(f"Updated: {report_path}")
