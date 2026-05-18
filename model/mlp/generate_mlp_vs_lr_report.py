"""Generate the MLP vs linear-regression family report from current CSVs.

Usage:
    python model/mlp/generate_mlp_vs_lr_report.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "mlp"
LR_BENCH_CSV = PROJECT_ROOT / "data" / "lr_benchmark.csv"
OUT_MD = SCRIPT_DIR / "mlp_vs_lr_report.md"

INPUT_WINDOWS = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
GRID_KEYS = ["input_window", "output_window"]
BASE_COLS = [
    "model_name",
    "input_window",
    "output_window",
    "MAE_train",
    "MAE_val",
    "MAE_test",
    "epochs",
    "n_params",
]


def load_results() -> pd.DataFrame:
    paths = sorted(
        path for path in RESULTS_DIR.glob("mlp_*.csv")
        if not path.name.endswith("_history.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No MLP summary CSV files found in {RESULTS_DIR}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [col for col in BASE_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frames.append(df)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["model_name", "input_window", "output_window"])
        .reset_index(drop=True)
    )


def pct_delta(delta: float, baseline: float) -> float:
    return 100.0 * delta / baseline


def md_table(df: pd.DataFrame, floatfmt: str = ".6f", index: bool = False) -> str:
    """Small Markdown table writer that avoids pandas' optional tabulate dependency."""
    table = df.reset_index() if index else df.copy()
    columns = [str(col) for col in table.columns]

    def fmt(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and floatfmt:
            return format(value, floatfmt)
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(all_results: pd.DataFrame, lr_df: pd.DataFrame) -> str:
    lr_comp = lr_df.rename(
        columns={"MAE_test": "MAE_test_lr", "MAE_train": "MAE_train_lr"}
    )
    comparison = all_results.merge(
        lr_comp[GRID_KEYS + ["MAE_test_lr"]],
        on=GRID_KEYS,
        how="left",
    )
    if comparison["MAE_test_lr"].isna().any():
        bad = comparison.loc[comparison["MAE_test_lr"].isna(), GRID_KEYS].drop_duplicates()
        raise ValueError(f"Missing LR benchmark for windows:\n{bad}")

    comparison["delta_vs_lr"] = comparison["MAE_test"] - comparison["MAE_test_lr"]
    comparison["pct_delta_vs_lr"] = (
        100 * comparison["delta_vs_lr"] / comparison["MAE_test_lr"]
    )

    idx = comparison.groupby(GRID_KEYS)["MAE_test"].idxmin()
    best_mlp = (
        comparison.loc[idx]
        .sort_values(GRID_KEYS)
        .reset_index(drop=True)
    )

    best_counts = best_mlp["model_name"].value_counts().to_dict()
    ranking = (
        comparison.groupby("model_name")
        .agg(
            mean_test=("MAE_test", "mean"),
            median_test=("MAE_test", "median"),
            best_test=("MAE_test", "min"),
            worst_test=("MAE_test", "max"),
            mean_delta_vs_lr=("delta_vs_lr", "mean"),
            wins_vs_lr=("delta_vs_lr", lambda s: int((s < 0).sum())),
            avg_epochs=("epochs", "mean"),
            avg_params=("n_params", "mean"),
        )
        .reset_index()
        .sort_values("mean_test")
    )
    ranking["best_mlp_cells"] = ranking["model_name"].map(best_counts).fillna(0).astype(int)
    ranking = ranking[
        [
            "model_name",
            "mean_test",
            "median_test",
            "best_test",
            "worst_test",
            "mean_delta_vs_lr",
            "wins_vs_lr",
            "best_mlp_cells",
            "avg_epochs",
            "avg_params",
        ]
    ]

    best_model = ranking.iloc[0]
    mean_lr = lr_df["MAE_test"].mean()
    best_mlp_mean = float(best_model["mean_test"])
    best_mlp_delta = best_mlp_mean - mean_lr
    best_cells_vs_lr = int((best_mlp["delta_vs_lr"] < 0).sum())

    best_per_window = best_mlp[
        GRID_KEYS + ["model_name", "MAE_test", "MAE_test_lr", "delta_vs_lr", "pct_delta_vs_lr"]
    ].copy()

    param_counts = (
        comparison.groupby(["model_name", "input_window"])["n_params"]
        .first()
        .unstack("input_window")
        .reindex(columns=INPUT_WINDOWS)
        .reset_index()
        .sort_values("model_name")
    )

    mae_matrix = (
        best_mlp.pivot(index="output_window", columns="input_window", values="MAE_test")
        .reindex(index=OUTPUT_WINDOWS, columns=INPUT_WINDOWS)
    )
    winner_matrix = (
        best_mlp.pivot(index="output_window", columns="input_window", values="model_name")
        .reindex(index=OUTPUT_WINDOWS, columns=INPUT_WINDOWS)
    )

    lines: list[str] = [
        "# MLP vs Linear Regression Report",
        "",
        "Generated from the current fixed MLP CSV files in `data/mlp/`.",
        "",
        "## Main Conclusion",
        "",
        f"The best current fixed MLP by mean test MAE is `{best_model['model_name']}`.",
        "",
        f"- Best fixed MLP mean test MAE: `{best_mlp_mean:.6f}`",
        f"- LR mean test MAE: `{mean_lr:.6f}`",
        f"- Mean delta vs LR: `{best_mlp_delta:+.6f}` ({pct_delta(best_mlp_delta, mean_lr):+.2f}%)",
        f"- Per-window best MLP beats LR: `{best_cells_vs_lr} / {len(best_mlp)}`",
        "",
        "## Model Ranking",
        "",
        md_table(ranking),
        "",
        "## Best MLP Per Window",
        "",
        md_table(best_per_window),
        "",
        "## Best MLP MAE Matrix",
        "",
        md_table(mae_matrix.reset_index()),
        "",
        "## Winning MLP Architecture Matrix",
        "",
        md_table(winner_matrix.reset_index(), floatfmt=""),
        "",
        "## Parameter Counts",
        "",
        md_table(param_counts, floatfmt=".0f"),
        "",
        "## Interpretation",
        "",
        f"- `{best_model['model_name']}` is the strongest fixed MLP by mean test MAE.",
        "- The per-window winner is not always the same architecture, so the global ranking and",
        "  the best-per-window table should be read together.",
        "- The newest `mlp_4x100_gelu_dropout_l2` rows include the internal Keras `Normalization`",
        "  layer parameters and the usable-checkpoint metadata, so its parameter count is slightly",
        "  higher than the older report version.",
        "- Use `model/mlp/99_compare_mlp_vs_lr.ipynb` for exploratory notebook views, and this",
        "  generator for the Markdown report consumed by `reports/competition/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    all_results = load_results()
    lr_df = pd.read_csv(LR_BENCH_CSV)
    report = build_report(all_results, lr_df)
    OUT_MD.write_text(report + "\n", encoding="utf-8")
    print(f"Report written -> {OUT_MD}")


if __name__ == "__main__":
    main()
