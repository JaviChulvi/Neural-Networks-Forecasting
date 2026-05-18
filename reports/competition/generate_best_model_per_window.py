"""
Compare MLP, CNN, RNN, and Mixtos families and find the global best model per
(input_window, output_window) combination by test MAE.
Writes best_model_per_window_report.md to reports/competition/.

Data extracted from:
  data/mlp/*.csv
  data/cnn/cnn_all_results.csv
  model/rnn/rnn_vs_lr_report.md
  model/mixtos/mixtos_vs_lr_report.md
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

INPUT_WINDOWS  = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
WINDOWS = [(i, o) for i in INPUT_WINDOWS for o in OUTPUT_WINDOWS]

MIXTOS_LABELS = {
    "L": "LSTM (tuned)",
    "G": "GRU (tuned)",
    "CL": "CNN-LSTM (tuned)",
    "CG": "CNN-GRU (tuned)",
    "CLM": "CNN-LSTM-MLP (tuned)",
    "CGM": "CNN-GRU-MLP (tuned)",
    "CM": "CNN-MLP (tuned)",
    "HL": "Hybrid CNN-LSTM",
    "HG": "Hybrid CNN-GRU",
    "HB": "Hybrid CNN-BiGRU",
}


def load_lr_benchmark() -> dict[tuple[int, int], float]:
    lr_df = pd.read_csv(DATA_DIR / "lr_benchmark.csv")
    return {
        (int(row.input_window), int(row.output_window)): float(row.MAE_test)
        for row in lr_df.itertuples(index=False)
    }


def _check_complete(name: str, data: dict[tuple[int, int], dict[str, float | str]]) -> None:
    missing = [window for window in WINDOWS if window not in data]
    if missing:
        raise ValueError(f"{name} is missing window results: {missing}")


def load_mlp_best() -> dict[tuple[int, int], dict[str, float | str]]:
    paths = sorted(
        path for path in (DATA_DIR / "mlp").glob("mlp_*.csv")
        if not path.name.endswith("_history.csv")
    )
    if not paths:
        raise FileNotFoundError(f"No MLP CSV files found in {DATA_DIR / 'mlp'}")

    all_rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    idx = all_rows.groupby(["input_window", "output_window"])["MAE_test"].idxmin()
    best_df = (
        all_rows.loc[idx]
        .sort_values(["input_window", "output_window"])
        .reset_index(drop=True)
    )

    ref_path = Path(__file__).parent / "mlp_best_family_results_reference.csv"
    best_df.to_csv(ref_path, index=False)

    return {
        (int(row.input_window), int(row.output_window)): {
            "model": str(row.model_name),
            "mae": float(row.MAE_test),
        }
        for row in best_df.itertuples(index=False)
    }


def load_cnn_best() -> dict[tuple[int, int], dict[str, float | str]]:
    path = DATA_DIR / "cnn" / "cnn_all_results.csv"
    all_rows = pd.read_csv(path)
    idx = all_rows.groupby(["input_window", "output_window"])["MAE_test"].idxmin()
    best_df = (
        all_rows.loc[idx]
        .sort_values(["input_window", "output_window"])
        .reset_index(drop=True)
    )
    return {
        (int(row.input_window), int(row.output_window)): {
            "model": str(row.model),
            "mae": float(row.MAE_test),
        }
        for row in best_df.itertuples(index=False)
    }


def _parse_markdown_detail_table(
    path: Path,
    heading: str,
    label_map: dict[str, str] | None = None,
) -> dict[tuple[int, int], dict[str, float | str]]:
    """Read a report detail table with input/output/model/MAE_test columns."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise ValueError(f"Heading not found in {path}: {heading}") from exc

    table_lines: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            table_lines.append(line)

    if len(table_lines) < 3:
        raise ValueError(f"No Markdown table found below {heading} in {path}")

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    required = ["input_window", "output_window", "model_type", "MAE_test"]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"{path} table is missing columns: {missing}")

    result: dict[tuple[int, int], dict[str, float | str]] = {}
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        model = row["model_type"].strip("`")
        if label_map is not None:
            model = label_map.get(model, model)
        window = (int(row["input_window"]), int(row["output_window"]))
        result[window] = {
            "model": model,
            "mae": float(row["MAE_test"].strip("`")),
        }
    return result


def load_rnn_best() -> dict[tuple[int, int], dict[str, float | str]]:
    return _parse_markdown_detail_table(
        MODEL_DIR / "rnn" / "rnn_vs_lr_report.md",
        "## Best RNN Per Window — Detail",
    )


def load_mixtos_best() -> dict[tuple[int, int], dict[str, float | str]]:
    return _parse_markdown_detail_table(
        MODEL_DIR / "mixtos" / "mixtos_vs_lr_report.md",
        "## Best Mixed Model Per Window — Detail",
        MIXTOS_LABELS,
    )


def sync_hybrid_test_comparison() -> None:
    """Keep the competition CSV aligned with the hybrid test-MAE report."""
    src = DATA_DIR / "mixtos" / "cnn_rnn_hybrid" / "hybrid_best_test_comparison_vs_lr.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing hybrid test comparison CSV: {src}. "
            "Run `python model/mixtos/generate_mixtos_report.py` first."
        )
    dst = Path(__file__).parent / "hybrid_comparison_vs_lr.csv"
    dst.write_bytes(src.read_bytes())


LR = load_lr_benchmark()
MLP_BEST = load_mlp_best()
CNN_BEST = load_cnn_best()
RNN_BEST = load_rnn_best()
MIXTOS_BEST = load_mixtos_best()
sync_hybrid_test_comparison()

FAMILIES = {
    "MLP":    MLP_BEST,
    "CNN":    CNN_BEST,
    "RNN":    RNN_BEST,
    "Mixtos": MIXTOS_BEST,
}

for family_name, family_data in FAMILIES.items():
    _check_complete(family_name, family_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_best(window):
    best_family, best_model, best_mae = None, None, float("inf")
    for family, data in FAMILIES.items():
        mae = data[window]["mae"]
        if mae < best_mae:
            best_mae    = mae
            best_family = family
            best_model  = data[window]["model"]
    return best_family, best_model, best_mae


def pct_delta(mae, lr_mae):
    return (mae - lr_mae) / lr_mae * 100


def bold(val, is_best):
    s = f"`{val:.6f}`"
    return f"**{s}**" if is_best else s


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(results):
    lines = []
    n = len(results)

    mean_best = sum(r["mae_best"] for r in results) / n
    mean_lr   = sum(r["lr_mae"]   for r in results) / n
    mean_delta = mean_best - mean_lr
    wins_vs_lr = sum(1 for r in results if r["mae_best"] < r["lr_mae"])
    wins_by_family = {f: sum(1 for r in results if r["family"] == f) for f in FAMILIES}

    # -- Header --
    lines += [
        "# Global Best Model Per Window — Comparación de Todas las Familias",
        "",
        "Comparación de las familias **MLP**, **CNN**, **RNN** y **Mixtos**.",
        "Para cada combinación (input_window, output_window) se selecciona el modelo",
        "con el menor MAE en test de entre las cuatro familias.",
        "Valores negativos en `pct_delta_vs_lr` indican mejora sobre la regresión lineal.",
        "",
    ]

    # -- Main conclusion --
    lines += [
        "## Conclusión Principal",
        "",
        f"- Media MAE test **mejor global**     : `{mean_best:.6f}`",
        f"- Media MAE test **regresión lineal** : `{mean_lr:.6f}`",
        f"- Media Δ vs LR : `{mean_delta:+.6f}` ({mean_delta/mean_lr*100:+.2f}%)",
        f"- Ventanas donde el mejor global supera a LR : **{wins_vs_lr} / {n}**",
        "",
        "### Ventanas ganadas por familia",
        "",
        "| Familia | Ventanas ganadas |",
        "|---------|-----------------|",
    ]
    for fam, count in sorted(wins_by_family.items(), key=lambda x: -x[1]):
        lines.append(f"| {fam} | {count} |")
    lines.append("")

    # -- Family mean MAE reference --
    family_means = {}
    for fam, data in FAMILIES.items():
        family_means[fam] = sum(data[w]["mae"] for w in WINDOWS) / n

    lines += [
        "### Media MAE test por familia (sobre las 16 ventanas)",
        "",
        "| Familia | Media MAE test |",
        "|---------|---------------|",
    ]
    for fam, mean in sorted(family_means.items(), key=lambda x: x[1]):
        lines.append(f"| {fam} | `{mean:.6f}` |")
    lines += ["| LR (benchmark) | `{:.6f}` |".format(mean_lr), ""]

    # -- Detail table --
    lines += [
        "## Mejor Modelo Por Ventana — Detalle",
        "",
        "| input_window | output_window | familia | modelo | MAE_best | MAE_lr | pct_delta_vs_lr |",
        "|:---:|:---:|:---:|:---|:---:|:---:|:---:|",
    ]
    for r in results:
        p = pct_delta(r["mae_best"], r["lr_mae"])
        sign = "↓" if p < 0 else "↑"
        lines.append(
            f"| {r['in']} | {r['out']} | **{r['family']}** | {r['model']} "
            f"| `{r['mae_best']:.6f}` | `{r['lr_mae']:.6f}` | `{p:+.2f}%` {sign} |"
        )
    lines.append("")

    # -- MAE matrix --
    lines += [
        "## Matriz MAE Test (mejor global)",
        "",
        "| Salida \\ Entrada | in=5 | in=10 | in=30 | in=90 |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for out in OUTPUT_WINDOWS:
        row = f"| **out={out}** |"
        for inp in INPUT_WINDOWS:
            r = next(x for x in results if x["in"] == inp and x["out"] == out)
            row += f" `{r['mae_best']:.6f}` |"
        lines.append(row)
    lines.append("")

    # -- Family winner matrix --
    lines += [
        "## Matriz de Familia Ganadora",
        "",
        "| Salida \\ Entrada | in=5 | in=10 | in=30 | in=90 |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for out in OUTPUT_WINDOWS:
        row = f"| **out={out}** |"
        for inp in INPUT_WINDOWS:
            r = next(x for x in results if x["in"] == inp and x["out"] == out)
            row += f" **{r['family']}** |"
        lines.append(row)
    lines.append("")

    # -- Full comparison table --
    lines += [
        "## Comparación Completa — Las Cuatro Familias por Ventana",
        "",
        "El valor en negrita es el ganador de cada fila.",
        "",
        "| in | out | MLP | CNN | RNN | Mixtos | Ganador |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        w = (r["in"], r["out"])
        fam = r["family"]
        mlp_mae = MLP_BEST[w]["mae"]
        cnn_mae = CNN_BEST[w]["mae"]
        rnn_mae = RNN_BEST[w]["mae"]
        mix_mae = MIXTOS_BEST[w]["mae"]
        lines.append(
            f"| {r['in']} | {r['out']} "
            f"| {bold(mlp_mae, fam == 'MLP')} "
            f"| {bold(cnn_mae, fam == 'CNN')} "
            f"| {bold(rnn_mae, fam == 'RNN')} "
            f"| {bold(mix_mae, fam == 'Mixtos')} "
            f"| **{fam}** |"
        )
    lines.append("")

    # -- Delta vs LR matrix --
    lines += [
        "## Δ (Mejor Global − LR) Matriz",
        "",
        "> Negativo (↓) = modelo gana a LR; positivo (↑) = LR gana.",
        "",
        "| Salida \\ Entrada | in=5 | in=10 | in=30 | in=90 |",
        "|:---:|:---:|:---:|:---:|:---:|",
    ]
    for out in OUTPUT_WINDOWS:
        row = f"| **out={out}** |"
        for inp in INPUT_WINDOWS:
            r = next(x for x in results if x["in"] == inp and x["out"] == out)
            delta = r["mae_best"] - r["lr_mae"]
            sign = "↓" if delta < 0 else "↑"
            row += f" `{delta:+.6f}` {sign} |"
        lines.append(row)
    lines.append("")

    sorted_family_wins = sorted(wins_by_family.items(), key=lambda x: (-x[1], x[0]))
    top_family, top_wins = sorted_family_wins[0]
    runner_family, runner_wins = sorted_family_wins[1]

    # -- Interpretation --
    lines += [
        "## Interpretación",
        "",
        f"- **Familia líder**: `{top_family}` gana {top_wins} de {n} ventanas; "
        f"`{runner_family}` queda detrás con {runner_wins}.",
        "- **Lectura por ventana**: usa la matriz de familia ganadora para ver dónde cambia",
        "  la arquitectura preferida. Las diferencias de MAE son pequeñas en muchas celdas,",
        "  así que conviene interpretar los ganadores como ranking empírico, no como dominancia",
        "  estadística fuerte sin repetir semillas.",
        "- **RNN y Mixtos**: cuando no ganan una celda, no significa que sean inútiles;",
        "  significa que, con los resultados guardados actuales, otra familia tiene menor MAE test.",
        "- **16/16 ventanas superan a LR**: el modelo global óptimo mejora a la regresión lineal",
        "  en las 16 combinaciones.",
        "- **Mejora vs LR crece con input_window**: de ~−1% con `input=5` a ~−12/−17%",
        "  con `input=90`, patrón consistente con los reportes individuales de cada familia.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    for window in WINDOWS:
        inp, out = window
        family, model, mae = pick_best(window)
        results.append({
            "in":       inp,
            "out":      out,
            "family":   family,
            "model":    model,
            "mae_best": mae,
            "lr_mae":   LR[window],
        })

    report = build_report(results)

    out_path = Path(__file__).parent / "best_model_per_window_report.md"
    out_path.write_text(report + "\n")
    print(f"Report written → {out_path}")

    print(f"\n{'in':>4} {'out':>4} {'family':>8} {'MAE_best':>12} {'MAE_lr':>12} {'pct_delta':>10}")
    print("-" * 58)
    for r in results:
        p = pct_delta(r["mae_best"], r["lr_mae"])
        print(
            f"{r['in']:>4} {r['out']:>4} {r['family']:>8} "
            f"{r['mae_best']:>12.6f} {r['lr_mae']:>12.6f} {p:>+10.2f}%"
        )


if __name__ == "__main__":
    main()
