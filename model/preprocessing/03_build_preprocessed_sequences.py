import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from util import RANDOM_SEED
except Exception:
    RANDOM_SEED = 42

DATA_IN = PROJECT_ROOT / "data" / "preprocessing"
SEQUENCES_DIR = DATA_IN / "sequences"
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_WINDOWS = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
BAR_TYPES = ["time", "count", "volume", "dollar"]


def make_sequences(returns: pd.DataFrame, input_window: int, output_window: int):
    """
    Builds supervised forecasting sequences from preprocessed bar returns.

    X contains the previous input_window bars of returns.
    y contains the average return over the next output_window bars.
    """
    values = returns.astype(float).values
    dates = returns.index

    X, y, target_dates = [], [], []
    max_i = len(values) - input_window - output_window + 1

    for i in range(max_i):
        x_i = values[i : i + input_window]
        y_i = values[i + input_window : i + input_window + output_window].mean(axis=0)
        X.append(x_i)
        y.append(y_i)
        target_dates.append(dates[i + input_window + output_window - 1])

    return np.array(X), np.array(y), np.array(target_dates)


def save_npz(path: Path, X_train, X_test, y_train, y_test, dates_train, dates_test, input_window, output_window, bar_type):
    np.savez_compressed(
        path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dates_train=dates_train.astype("datetime64[ns]"),
        dates_test=dates_test.astype("datetime64[ns]"),
        input_window_size=np.array(input_window),
        output_window_size=np.array(output_window),
        bar_type=np.array(bar_type),
    )


def main():
    rows = []

    for bar_type in BAR_TYPES:
        returns_path = DATA_IN / f"{bar_type}_bars_returns.parquet"

        if not returns_path.exists():
            print(f"SKIP missing returns file: {returns_path}")
            continue

        returns = pd.read_parquet(returns_path).dropna(how="all")
        returns = returns.dropna(axis=1, how="all")
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        bar_out_dir = SEQUENCES_DIR / bar_type
        bar_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {bar_type.upper()} BARS ===")
        print("returns shape:", returns.shape)
        print("date range:", returns.index.min(), "->", returns.index.max())

        for input_window in INPUT_WINDOWS:
            for output_window in OUTPUT_WINDOWS:
                min_required = input_window + output_window + 2

                if len(returns) < min_required:
                    rows.append({
                        "bar_type": bar_type,
                        "input_window": input_window,
                        "output_window": output_window,
                        "status": "skipped_not_enough_data",
                        "n_bars_available": len(returns),
                        "n_samples_total": 0,
                        "n_train": 0,
                        "n_test": 0,
                        "n_assets": returns.shape[1],
                        "start_date": returns.index.min(),
                        "end_date": returns.index.max(),
                        "sequence_file": "",
                    })
                    continue

                X, y, target_dates = make_sequences(returns, input_window, output_window)

                if len(X) == 0:
                    status = "skipped_no_sequences"
                    n_train = n_test = 0
                    seq_rel_path = ""
                else:
                    idx = np.arange(len(X))
                    train_idx, test_idx = train_test_split(
                        idx,
                        test_size=0.10,
                        random_state=RANDOM_SEED,
                        shuffle=True,
                    )

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    dates_train, dates_test = target_dates[train_idx], target_dates[test_idx]

                    seq_path = bar_out_dir / f"{bar_type}_input{input_window}_output{output_window}.npz"
                    save_npz(
                        seq_path,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        dates_train,
                        dates_test,
                        input_window,
                        output_window,
                        bar_type,
                    )

                    status = "ok"
                    n_train = len(X_train)
                    n_test = len(X_test)
                    seq_rel_path = str(seq_path.relative_to(PROJECT_ROOT))

                rows.append({
                    "bar_type": bar_type,
                    "input_window": input_window,
                    "output_window": output_window,
                    "status": status,
                    "n_bars_available": len(returns),
                    "n_samples_total": len(X),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_assets": returns.shape[1],
                    "start_date": returns.index.min(),
                    "end_date": returns.index.max(),
                    "sequence_file": seq_rel_path,
                })

                print(
                    f"{bar_type:6s} input={input_window:2d} output={output_window:2d} "
                    f"samples={len(X):5d} train={n_train:5d} test={n_test:4d}"
                )

    summary = pd.DataFrame(rows)
    summary_path = DATA_IN / "preprocessed_sequences_summary.csv"
    summary.to_csv(summary_path, index=False)

    matrix_path = DATA_IN / "preprocessed_sequences_sample_matrix.csv"
    matrix = summary.pivot_table(
        index=["bar_type", "input_window"],
        columns="output_window",
        values="n_samples_total",
        aggfunc="first",
    )
    matrix.to_csv(matrix_path)

    print("\nSaved:", summary_path)
    print("Saved:", matrix_path)
    print("\nSequence summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
