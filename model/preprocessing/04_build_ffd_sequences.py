"""
04_build_ffd_sequences.py

Builds supervised forecasting sequences using Fractional Differencing (FFD)
instead of log returns, following MLdP Chapter 5 (López de Prado).

For each bar type the script:
  1. Reads the close-price parquet produced by 02_activity_bars_yahoo.ipynb.
  2. Applies log transform to stabilise variance.
  3. Finds, per asset, the minimum d in [0, 1] that passes the ADF stationarity
     test (same approach as preprocesadoDatos.ipynb).
  4. Applies FFD with that d, preserving as much memory as possible.
  5. Builds (X, y) sequences with the same window sizes as script 03.
  6. Saves .npz files under data/preprocessing/sequences_ffd/.

Output layout mirrors 03_build_preprocessed_sequences.py so downstream model
code can swap between log-return and FFD sequences with minimal changes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from util import RANDOM_SEED
except Exception:
    RANDOM_SEED = 42

DATA_IN = PROJECT_ROOT / "data" / "preprocessing"
SEQUENCES_DIR = DATA_IN / "sequences_ffd"
SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_WINDOWS = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
BAR_TYPES = ["time", "count", "volume", "dollar"]

# Maps bar_type name -> close parquet path (relative to PROJECT_ROOT).
# "raw" points to the original daily close prices with no bar aggregation.
CLOSE_PATHS: dict[str, Path] = {
    **{bt: DATA_IN / f"{bt}_bars_close.parquet" for bt in BAR_TYPES},
    "raw": PROJECT_ROOT / "data" / "precios_close.parquet",
}

D_VALUES = np.arange(0.0, 1.05, 0.1)
FFD_WEIGHT_THRESHOLD = 1e-5
ADF_PVALUE_THRESHOLD = 0.05
ADF_MAXLAG = 10
ADF_MIN_OBS = 50


# ---------------------------------------------------------------------------
# FFD core (ported from preprocesadoDatos.ipynb, MLdP Cap. 5)
# ---------------------------------------------------------------------------

def get_weights_ffd(d: float, thres: float = FFD_WEIGHT_THRESHOLD) -> np.ndarray:
    """Compute FFD weights until |weight| < thres (fixed-width window)."""
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1], dtype=np.float64)


def _apply_ffd_1d(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Apply FFD weights to a 1-D array using a vectorised sliding window."""
    width = len(w)
    n = len(values)
    if n < width:
        return np.full(n, np.nan)

    # Build a (T-width+1, width) view without copying data
    from numpy.lib.stride_tricks import as_strided
    shape = (n - width + 1, width)
    strides = (values.strides[0], values.strides[0])
    windows = as_strided(values, shape=shape, strides=strides)
    valid = windows @ w  # shape: (T-width+1,)

    result = np.empty(n)
    result[:width - 1] = np.nan
    result[width - 1:] = valid
    return result


def fractional_difference(df: pd.DataFrame, d: float, thres: float = FFD_WEIGHT_THRESHOLD) -> pd.DataFrame:
    """Apply FFD independently to each column of df, returning a DataFrame."""
    w = get_weights_ffd(d, thres)
    out = {}
    for col in df.columns:
        s = df[col].ffill().dropna()
        result = _apply_ffd_1d(s.to_numpy(), w)
        out[col] = pd.Series(result, index=s.index, name=col)
    return pd.DataFrame(out).dropna(how="all")


def find_min_d(
    series: pd.Series,
    d_values: np.ndarray = D_VALUES,
    thres: float = FFD_WEIGHT_THRESHOLD,
    pvalue_threshold: float = ADF_PVALUE_THRESHOLD,
    maxlag: int = ADF_MAXLAG,
) -> float:
    """
    Return the smallest d in d_values for which the FFD series passes ADF.
    Falls back to d=1.0 (full differencing) if none qualifies.
    """
    s = series.ffill().dropna().to_numpy()

    for d in d_values:
        w = get_weights_ffd(d, thres)
        fd = _apply_ffd_1d(s, w)
        fd = fd[~np.isnan(fd)]

        if len(fd) < ADF_MIN_OBS or np.unique(fd).size < 5:
            continue

        try:
            _, p_value, *_ = adfuller(fd, maxlag=maxlag, autolag=None)
            if p_value < pvalue_threshold:
                return float(d)
        except Exception:
            continue

    return 1.0  # fallback: full differencing


def build_ffd_features(log_close: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Find the optimal d per asset, apply FFD, and return the aligned DataFrame
    together with a dict mapping each column name to its d value.
    """
    d_map: dict[str, float] = {}
    ffd_cols: dict[str, pd.Series] = {}

    n_assets = log_close.shape[1]
    for i, col in enumerate(log_close.columns, 1):
        print(f"  [{i}/{n_assets}] {col}", end="  ")
        s = log_close[col].ffill().dropna()
        d_opt = find_min_d(s)
        d_map[col] = d_opt
        print(f"d={d_opt:.1f}")

        w = get_weights_ffd(d_opt)
        result = _apply_ffd_1d(s.to_numpy(), w)
        ffd_cols[col] = pd.Series(result, index=s.index, name=col)

    ffd_df = pd.DataFrame(ffd_cols).dropna(how="all")
    return ffd_df, d_map


# ---------------------------------------------------------------------------
# Sequence building (same logic as 03_build_preprocessed_sequences.py)
# ---------------------------------------------------------------------------

def make_sequences(ffd: pd.DataFrame, input_window: int, output_window: int):
    values = ffd.astype(float).values
    dates = ffd.index
    X, y, target_dates = [], [], []
    max_i = len(values) - input_window - output_window + 1
    for i in range(max_i):
        x_i = values[i : i + input_window]
        y_i = values[i + input_window : i + input_window + output_window].mean(axis=0)
        X.append(x_i)
        y.append(y_i)
        target_dates.append(dates[i + input_window + output_window - 1])
    return np.array(X), np.array(y), np.array(target_dates)


def save_npz(
    path: Path,
    X_train, X_test, y_train, y_test,
    dates_train, dates_test,
    input_window: int, output_window: int, bar_type: str,
) -> None:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(bar_types: list[str] | None = None) -> None:
    """
    Parameters
    ----------
    bar_types : list of str, optional
        Subset of CLOSE_PATHS keys to process. Defaults to all available types.
    """
    rows = []
    paths = {k: v for k, v in CLOSE_PATHS.items() if bar_types is None or k in bar_types}

    for bar_type, close_path in paths.items():
        if not close_path.exists():
            print(f"SKIP missing close file: {close_path}")
            continue

        close = pd.read_parquet(close_path)
        close = close.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        close = close.dropna(axis=1, how="all")

        print(f"\n=== {bar_type.upper()} BARS ===")
        print(f"close shape: {close.shape}  |  date range: {close.index.min()} -> {close.index.max()}")

        log_close = np.log(close)

        print("Finding optimal d per asset and applying FFD...")
        ffd, d_map = build_ffd_features(log_close)
        ffd = ffd.replace([np.inf, -np.inf], np.nan).dropna(how="all")

        d_vals = list(d_map.values())
        print(f"d summary: min={min(d_vals):.1f}  max={max(d_vals):.1f}  mean={np.mean(d_vals):.2f}")
        print(f"FFD shape after dropna: {ffd.shape}")

        # Save d values for traceability and reproducibility
        d_df = pd.DataFrame.from_dict(d_map, orient="index", columns=["d_optimal"])
        d_df.to_csv(DATA_IN / f"{bar_type}_bars_ffd_d_values.csv")

        bar_out_dir = SEQUENCES_DIR / bar_type
        bar_out_dir.mkdir(parents=True, exist_ok=True)

        for input_window in INPUT_WINDOWS:
            for output_window in OUTPUT_WINDOWS:
                min_required = input_window + output_window + 2

                if len(ffd) < min_required:
                    rows.append({
                        "bar_type": bar_type,
                        "input_window": input_window,
                        "output_window": output_window,
                        "status": "skipped_not_enough_data",
                        "n_bars_available": len(ffd),
                        "n_samples_total": 0,
                        "n_train": 0,
                        "n_test": 0,
                        "n_assets": ffd.shape[1],
                        "mean_d": np.mean(d_vals),
                        "start_date": ffd.index.min(),
                        "end_date": ffd.index.max(),
                        "sequence_file": "",
                    })
                    continue

                X, y, target_dates = make_sequences(ffd, input_window, output_window)

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
                        shuffle=False,
                    )
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    dates_train = target_dates[train_idx]
                    dates_test = target_dates[test_idx]

                    seq_path = bar_out_dir / f"{bar_type}_ffd_input{input_window}_output{output_window}.npz"
                    save_npz(
                        seq_path,
                        X_train, X_test, y_train, y_test,
                        dates_train, dates_test,
                        input_window, output_window, bar_type,
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
                    "n_bars_available": len(ffd),
                    "n_samples_total": len(X),
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_assets": ffd.shape[1],
                    "mean_d": np.mean(d_vals),
                    "start_date": ffd.index.min(),
                    "end_date": ffd.index.max(),
                    "sequence_file": seq_rel_path,
                })

                print(
                    f"  {bar_type:6s} input={input_window:2d} output={output_window:2d} "
                    f"samples={len(X):5d} train={n_train:5d} test={n_test:4d}"
                )

    summary = pd.DataFrame(rows)
    summary_path = DATA_IN / "ffd_sequences_summary.csv"
    summary.to_csv(summary_path, index=False)

    matrix_path = DATA_IN / "ffd_sequences_sample_matrix.csv"
    matrix = summary.pivot_table(
        index=["bar_type", "input_window"],
        columns="output_window",
        values="n_samples_total",
        aggfunc="first",
    )
    matrix.to_csv(matrix_path)

    print("\nSaved:", summary_path)
    print("Saved:", matrix_path)
    print("\nFFD sequence summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FFD sequences for one or more bar types.")
    parser.add_argument(
        "--bar-types",
        nargs="+",
        choices=list(CLOSE_PATHS.keys()),
        default=None,
        metavar="TYPE",
        help=f"Bar types to process. Choices: {list(CLOSE_PATHS.keys())}. Defaults to all.",
    )
    args = parser.parse_args()
    main(bar_types=args.bar_types)
