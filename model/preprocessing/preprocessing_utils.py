from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "util.py").exists():
    PROJECT_ROOT = next(p for p in PROJECT_ROOT.parents if (p / "util.py").exists())

DATA_OUT = PROJECT_ROOT / "data" / "preprocessing"
PLOTS_DIR = DATA_OUT / "plots"
DATA_OUT.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def get_forecasting_tickers() -> List[str]:
    prices_path = PROJECT_ROOT / "data" / "precios_close.parquet"
    if not prices_path.exists():
        raise FileNotFoundError("data/precios_close.parquet not found")
    prices = pd.read_parquet(prices_path)
    return list(prices.columns)


def get_forecasting_date_range() -> Tuple[pd.Timestamp, pd.Timestamp]:
    prices_path = PROJECT_ROOT / "data" / "precios_close.parquet"
    prices = pd.read_parquet(prices_path)
    return pd.Timestamp(prices.index.min()), pd.Timestamp(prices.index.max())


def flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo MultiIndex columns to Field__Ticker."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [f"{str(a)}__{str(b)}" for a, b in out.columns]
    return out


def unflatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Field__Ticker columns back to a MultiIndex when possible."""
    if isinstance(df.columns, pd.MultiIndex):
        return df
    pairs = []
    ok = True
    for c in df.columns:
        if "__" not in c:
            ok = False
            break
        field, ticker = c.split("__", 1)
        pairs.append((field, ticker))
    if not ok:
        return df
    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(pairs, names=["Price", "Ticker"])
    return out


def extract_field(ohlcv: pd.DataFrame, field: str) -> pd.DataFrame:
    """Extract one OHLCV field from flattened or MultiIndex Yahoo data."""
    if isinstance(ohlcv.columns, pd.MultiIndex):
        if field not in ohlcv.columns.get_level_values(0):
            raise KeyError(f"Field {field} not found in OHLCV columns")
        return ohlcv[field].copy()

    prefix = f"{field}__"
    cols = [c for c in ohlcv.columns if str(c).startswith(prefix)]
    if not cols:
        raise KeyError(f"Field {field} not found in flattened OHLCV columns")
    out = ohlcv[cols].copy()
    out.columns = [c.split("__", 1)[1] for c in cols]
    return out


def download_yahoo_ohlcv(tickers: List[str], start: str, end: str, output_path: Path | None = None) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    raw = flatten_yahoo_columns(raw)
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    if output_path is None:
        output_path = DATA_OUT / "yahoo_ohlcv.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output_path)
    return raw


def compute_universe_activity(ohlcv: pd.DataFrame) -> pd.DataFrame:
    close = extract_field(ohlcv, "Close").sort_index()
    volume = extract_field(ohlcv, "Volume").reindex(close.index)
    volume = volume.reindex(columns=close.columns)

    dollar_volume = close * volume

    activity = pd.DataFrame(index=close.index)
    activity["total_volume"] = volume.sum(axis=1, skipna=True)
    activity["total_dollar"] = dollar_volume.sum(axis=1, skipna=True)
    activity["valid_close_assets"] = close.notna().sum(axis=1)
    activity["valid_volume_assets"] = volume.notna().sum(axis=1)
    return activity


def make_bar_end_indices_by_count(n_rows: int, count_threshold: int) -> List[int]:
    if count_threshold <= 0:
        raise ValueError("count_threshold must be positive")
    end_indices = list(range(count_threshold - 1, n_rows, count_threshold))
    if not end_indices or end_indices[-1] != n_rows - 1:
        end_indices.append(n_rows - 1)
    return sorted(set(end_indices))


def make_bar_end_indices_by_threshold(values: pd.Series, threshold: float) -> List[int]:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    end_indices = []
    acc = 0.0
    for i, v in enumerate(values.fillna(0.0).to_numpy()):
        acc += float(v)
        if acc >= threshold:
            end_indices.append(i)
            acc = 0.0
    if not end_indices or end_indices[-1] != len(values) - 1:
        end_indices.append(len(values) - 1)
    return sorted(set(end_indices))


def close_at_bar_ends(close: pd.DataFrame, end_indices: List[int]) -> pd.DataFrame:
    out = close.iloc[end_indices].copy()
    out.index.name = "Date"
    return out


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff().dropna(how="all")


def bar_durations(close: pd.DataFrame, end_indices: List[int]) -> pd.Series:
    dates = close.index[end_indices]
    durations = pd.Series(dates, index=dates).diff().dt.days
    durations.iloc[0] = np.nan
    return durations


def build_activity_bars(ohlcv: pd.DataFrame, target_bars: int = 1000) -> Dict[str, Dict[str, object]]:
    close = extract_field(ohlcv, "Close").sort_index()
    activity = compute_universe_activity(ohlcv).reindex(close.index)

    n_rows = len(close)
    if target_bars <= 0:
        raise ValueError("target_bars must be positive")

    count_threshold = max(1, int(np.floor(n_rows / target_bars)))
    volume_threshold = float(activity["total_volume"].sum() / target_bars)
    dollar_threshold = float(activity["total_dollar"].sum() / target_bars)

    definitions = {
        "time": {
            "description": "Daily time bars",
            "threshold_type": "calendar_day",
            "threshold": 1.0,
            "end_indices": list(range(n_rows)),
        },
        "count": {
            "description": "Count bars: proxy for tick bars using daily observations",
            "threshold_type": "n_days",
            "threshold": float(count_threshold),
            "end_indices": make_bar_end_indices_by_count(n_rows, count_threshold),
        },
        "volume": {
            "description": "Daily volume bars using aggregated universe volume",
            "threshold_type": "total_volume",
            "threshold": volume_threshold,
            "end_indices": make_bar_end_indices_by_threshold(activity["total_volume"], volume_threshold),
        },
        "dollar": {
            "description": "Daily dollar bars using aggregated universe dollar volume",
            "threshold_type": "total_dollar",
            "threshold": dollar_threshold,
            "end_indices": make_bar_end_indices_by_threshold(activity["total_dollar"], dollar_threshold),
        },
    }

    bars = {}
    for name, info in definitions.items():
        end_indices = info["end_indices"]
        bars_close = close_at_bar_ends(close, end_indices)
        bars_returns = log_returns(bars_close)
        durations = bar_durations(close, end_indices)
        bars[name] = {
            **info,
            "close": bars_close,
            "returns": bars_returns,
            "durations": durations,
        }
    return bars


def summarize_bars(bars: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for name, item in bars.items():
        close = item["close"]
        returns = item["returns"]
        durations = item["durations"]
        values = returns.to_numpy().ravel()
        values = values[~np.isnan(values)]
        rows.append({
            "bar_type": name,
            "description": item["description"],
            "threshold_type": item["threshold_type"],
            "threshold": item["threshold"],
            "n_bars": len(close),
            "start_date": close.index.min(),
            "end_date": close.index.max(),
            "mean_days_per_bar": float(durations.mean(skipna=True)) if len(durations) else np.nan,
            "median_days_per_bar": float(durations.median(skipna=True)) if len(durations) else np.nan,
            "mean_abs_return": float(np.mean(np.abs(values))) if len(values) else np.nan,
            "return_std": float(np.std(values)) if len(values) else np.nan,
            "return_skew": float(pd.Series(values).skew()) if len(values) else np.nan,
            "return_kurtosis": float(pd.Series(values).kurtosis()) if len(values) else np.nan,
        })
    return pd.DataFrame(rows)


def save_bars(bars: Dict[str, Dict[str, object]], output_dir: Path | None = None) -> None:
    if output_dir is None:
        output_dir = DATA_OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, item in bars.items():
        item["close"].to_parquet(output_dir / f"{name}_bars_close.parquet")
        item["returns"].to_parquet(output_dir / f"{name}_bars_returns.parquet")
        pd.DataFrame({"duration_days": item["durations"]}).to_csv(output_dir / f"{name}_bars_durations.csv")


def plot_bar_counts(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(summary["bar_type"], summary["n_bars"])
    plt.title("Number of bars by preprocessing method")
    plt.xlabel("Bar type")
    plt.ylabel("Number of bars")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_return_distributions(bars: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for name, item in bars.items():
        values = item["returns"].to_numpy().ravel()
        values = values[~np.isnan(values)]
        if len(values):
            plt.hist(values, bins=80, alpha=0.35, density=True, label=name)
    plt.title("Return distribution by bar type")
    plt.xlabel("Log return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_bar_durations(bars: Dict[str, Dict[str, object]], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    for name, item in bars.items():
        if name == "time":
            continue
        durations = item["durations"].dropna().to_numpy()
        if len(durations):
            plt.hist(durations, bins=50, alpha=0.35, density=True, label=name)
    plt.title("Distribution of bar durations")
    plt.xlabel("Days per bar")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def markdown_table(df: pd.DataFrame) -> str:
    """Small Markdown table writer that avoids pandas' optional tabulate dependency."""
    columns = [str(col) for col in df.columns]

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def create_preprocessing_report(summary: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = PROJECT_ROOT / "model" / "preprocessing" / "preprocessing_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Financial data preprocessing report",
        "",
        "## Objective",
        "",
        "This report applies the financial preprocessing ideas from the first workshop of block 3 to the forecasting dataset.",
        "",
        "The original repository stores adjusted close prices and log returns. To build activity-based bars, daily OHLCV data is re-downloaded from Yahoo Finance for the same universe of assets.",
        "",
        "## Important limitation",
        "",
        "Yahoo Finance does not provide transaction-level trades in this dataset. Therefore, real tick bars cannot be constructed. Count bars are used as a daily proxy for tick bars, while volume bars and dollar bars are built using aggregated daily volume and dollar volume across the full asset universe.",
        "",
        "## Generated bar types",
        "",
        "- Time bars: original daily observations.",
        "- Count bars: every fixed number of daily observations.",
        "- Volume bars: days are grouped until a threshold of aggregated universe volume is reached.",
        "- Dollar bars: days are grouped until a threshold of aggregated universe dollar volume is reached.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Output files",
        "",
        "Generated datasets are stored under `data/preprocessing/`.",
        "",
        "The outputs include close prices, log returns, bar durations, summary statistics and plots for the different bar types.",
        "",
    ]
    output_path.write_text("\n".join(lines))
    return output_path
