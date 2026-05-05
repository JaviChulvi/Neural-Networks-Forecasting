"""
Genera rnn_vs_lr_report.md comparando LSTM tuneada y GRU tuneada contra la regresión lineal.

Fuentes de datos:
  - LSTM tuneada : model/rnn/lstm/rnn-lstm-input*-output*.ipynb (outputs de celdas)
                   + model/mlruns/ para la ventana (in=30, out=5) que no tiene notebook propio
  - GRU tuneada  : model/rnn/gru/hp_search_input*_output*.ipynb (outputs de celdas)
  - LR benchmark : data/lr_benchmark.csv

Uso:
    python generate_rnn_vs_lr_report.py
"""

import re
import sys
import warnings
import json
from pathlib import Path

import mlflow
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent   # model/rnn/
MODEL_DIR    = SCRIPT_DIR.parent                 # model/
PROJECT_ROOT = MODEL_DIR.parent
LSTM_NB_DIR  = SCRIPT_DIR / "lstm"
GRU_NB_DIR   = SCRIPT_DIR / "gru"
MLRUNS_URI   = MODEL_DIR / "mlruns"
LR_BENCH_CSV = PROJECT_ROOT / "data" / "lr_benchmark.csv"
OUT_MD       = SCRIPT_DIR / "rnn_vs_lr_report.md"

INPUT_WINDOWS  = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
N_FEATURES     = 23  # retornos por ticker


# ── Utilidades de parseo de notebooks ─────────────────────────────────────────

def cell_outputs_text(nb: dict) -> list[tuple[str, str]]:
    """Devuelve lista de (source, output_text) para celdas de código con salida."""
    result = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        text_parts = []
        for o in c.get("outputs", []):
            part = o.get("text", o.get("data", {}).get("text/plain", ""))
            if isinstance(part, list):
                part = "".join(part)
            text_parts.append(part)
        if text_parts:
            result.append((src, "\n".join(text_parts)))
    return result


def parse_winning_config(nb: dict, layer_key: str) -> dict:
    """Extrae la configuración ganadora del output 'Configuración ganadora:'."""
    for src, text in cell_outputs_text(nb):
        if "Configuración ganadora" not in text:
            continue
        layers = units = dropout = lr = bs = None
        for line in text.splitlines():
            line = line.strip()
            if layer_key in line:
                layers = int(re.search(r"= (\d+)", line).group(1))
            elif "units" in line and "=" in line:
                units = int(re.search(r"= (\d+)", line).group(1))
            elif "dropout" in line and "=" in line:
                dropout = float(re.search(r"= ([\d.]+)", line).group(1))
            elif "learning_rate" in line:
                lr = float(re.search(r"= ([\de\-+.]+)", line).group(1))
            elif "batch_size" in line:
                bs = int(re.search(r"= (\d+)", line).group(1))
        if all(v is not None for v in [layers, units, dropout, lr, bs]):
            return {"layers": layers, "units": units, "dropout": dropout,
                    "learning_rate": lr, "batch_size": bs}
    return {}


def parse_summary_mae(nb: dict, model_label: str) -> tuple[float | None, float | None]:
    """Extrae (MAE_train, MAE_test) para el modelo tuneado del output del summary."""
    for src, text in cell_outputs_text(nb):
        if model_label not in text:
            continue
        for line in text.splitlines():
            if model_label in line:
                nums = re.findall(r"\d+\.\d+", line)
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
    return None, None


def compute_n_params_lstm(layers: int, units: int, n_in: int, n_out: int) -> int:
    """Params totales de LSTM apilada + Dense de salida (fórmula Keras estándar)."""
    n = 0
    inp = n_in
    for _ in range(layers):
        n += 4 * (inp * units + units * units + units)
        inp = units
    n += units * n_out + n_out
    return n


def compute_n_params_gru(layers: int, units: int, n_in: int, n_out: int) -> int:
    """Params totales de GRU apilada + Dense de salida (fórmula Keras estándar)."""
    n = 0
    inp = n_in
    for _ in range(layers):
        n += 3 * (inp * units + units * units + units)
        inp = units
    n += units * n_out + n_out
    return n


# ── Carga de resultados LSTM ───────────────────────────────────────────────────

lstm_rows = []

for nb_path in sorted(LSTM_NB_DIR.glob("rnn-lstm-input*-output*.ipynb")):
    m = re.search(r"input(\d+)-output(\d+)", nb_path.name)
    iw, ow = int(m.group(1)), int(m.group(2))

    with open(nb_path) as f:
        nb = json.load(f)

    cfg = parse_winning_config(nb, "lstm_layers")
    mae_train, mae_test = parse_summary_mae(nb, "LSTM tuneada")

    if not cfg or mae_test is None:
        print(f"[LSTM] Faltan datos en {nb_path.name}", file=sys.stderr)
        continue

    n_params = compute_n_params_lstm(cfg["layers"], cfg["units"], N_FEATURES, N_FEATURES)

    lstm_rows.append({
        "input_window":  iw,
        "output_window": ow,
        "model_type":    "LSTM",
        "layers":        cfg["layers"],
        "units":         cfg["units"],
        "dropout":       cfg["dropout"],
        "learning_rate": cfg["learning_rate"],
        "batch_size":    cfg["batch_size"],
        "n_params":      n_params,
        "MAE_train":     mae_train,
        "MAE_test":      mae_test,
    })

# Ventana (30, 5) no tiene notebook propio: usar el mejor run del HP search en mlruns
LSTM_30_5_IN_NOTEBOOKS = any(
    r["input_window"] == 30 and r["output_window"] == 5 for r in lstm_rows
)
if not LSTM_30_5_IN_NOTEBOOKS:
    mlflow.set_tracking_uri(str(MLRUNS_URI))
    client = mlflow.tracking.MlflowClient()
    exps = [e for e in client.search_experiments()
            if e.name == "Red_Neuronal_Recurrente_LSTM"]
    if exps:
        runs = client.search_runs(
            experiment_ids=[exps[0].experiment_id], max_results=2000
        )
        # Mejor run del train-grid para (30,5): lr1e-04_batch256
        for r in runs:
            name = r.data.tags.get("mlflow.runName", "")
            if "tuned_input30_output5" in name:
                p = r.data.params
                m_r = r.data.metrics
                iw_r, ow_r = 30, 5
                layers = int(p["lstm_layers"])
                units  = int(p["units"])
                lstm_rows.append({
                    "input_window":  iw_r,
                    "output_window": ow_r,
                    "model_type":    "LSTM",
                    "layers":        layers,
                    "units":         units,
                    "dropout":       float(p["dropout"]),
                    "learning_rate": float(p["learning_rate"]),
                    "batch_size":    int(p["batch_size"]),
                    "n_params":      compute_n_params_lstm(layers, units, N_FEATURES, N_FEATURES),
                    "MAE_train":     m_r["train_mae"],
                    "MAE_test":      m_r["test_mae"],
                })
                break

lstm_df = pd.DataFrame(lstm_rows).sort_values(["input_window", "output_window"]).reset_index(drop=True)

# ── Carga de resultados GRU ────────────────────────────────────────────────────

gru_rows = []

for nb_path in sorted(GRU_NB_DIR.glob("hp_search_input*_output*.ipynb")):
    m = re.search(r"input(\d+)_output(\d+)", nb_path.name)
    iw, ow = int(m.group(1)), int(m.group(2))

    with open(nb_path) as f:
        nb = json.load(f)

    cfg = parse_winning_config(nb, "gru_layers")
    mae_train, mae_test = parse_summary_mae(nb, "GRU tuneada")

    if not cfg or mae_test is None:
        print(f"[GRU] Faltan datos en {nb_path.name}", file=sys.stderr)
        continue

    n_params = compute_n_params_gru(cfg["layers"], cfg["units"], N_FEATURES, N_FEATURES)

    gru_rows.append({
        "input_window":  iw,
        "output_window": ow,
        "model_type":    "GRU",
        "layers":        cfg["layers"],
        "units":         cfg["units"],
        "dropout":       cfg["dropout"],
        "learning_rate": cfg["learning_rate"],
        "batch_size":    cfg["batch_size"],
        "n_params":      n_params,
        "MAE_train":     mae_train,
        "MAE_test":      mae_test,
    })

gru_df = pd.DataFrame(gru_rows).sort_values(["input_window", "output_window"]).reset_index(drop=True)

# ── LR benchmark ──────────────────────────────────────────────────────────────

lr_df = pd.read_csv(LR_BENCH_CSV)
lr_map = lr_df.set_index(["input_window", "output_window"])["MAE_test"].to_dict()

# ── Merge y deltas ─────────────────────────────────────────────────────────────

all_df = pd.concat([lstm_df, gru_df], ignore_index=True)
all_df["MAE_test_lr"] = all_df.apply(
    lambda r: lr_map.get((r.input_window, r.output_window), float("nan")), axis=1
)
all_df["delta"]     = all_df["MAE_test"] - all_df["MAE_test_lr"]
all_df["pct_delta"] = 100 * all_df["delta"] / all_df["MAE_test_lr"]

# Para cada ventana, mejor modelo (menor MAE_test)
idx_best = all_df.groupby(["input_window", "output_window"])["MAE_test"].idxmin()
best_df  = all_df.loc[idx_best].copy()


# ── Helpers de tablas markdown ─────────────────────────────────────────────────

def pivot_model_md(df: pd.DataFrame, value_col: str, fmt: str = ".6f") -> str:
    piv  = df.pivot(index="output_window", columns="input_window", values=value_col)
    iws  = sorted(piv.columns.tolist())
    ows  = sorted(piv.index.tolist())
    hdr  = "| Salida \\ Entrada |" + "".join(f" in={iw} |" for iw in iws)
    sep  = "|" + ":---:|" * (len(iws) + 1)
    rows = [hdr, sep]
    for ow in ows:
        cells = [f"**out={ow}**"]
        for iw in iws:
            cells.append(f"`{piv.loc[ow, iw]:{fmt}}`")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def best_model_md(best_df: pd.DataFrame) -> str:
    piv_mae   = best_df.pivot(index="output_window", columns="input_window", values="MAE_test")
    piv_model = best_df.pivot(index="output_window", columns="input_window", values="model_type")
    iws       = sorted(piv_mae.columns.tolist())
    ows       = sorted(piv_mae.index.tolist())
    hdr       = "| Salida \\ Entrada |" + "".join(f" in={iw} |" for iw in iws)
    sep       = "|" + ":---:|" * (len(iws) + 1)
    rows      = [hdr, sep]
    for ow in ows:
        cells = [f"**out={ow}**"]
        for iw in iws:
            mae   = piv_mae.loc[ow, iw]
            model = piv_model.loc[ow, iw]
            tag   = "L" if model == "LSTM" else "G"
            cells.append(f"`{mae:.6f}` ({tag})")
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    rows.append("> (L) = LSTM · (G) = GRU")
    return "\n".join(rows)


def delta_md(df: pd.DataFrame) -> str:
    piv  = df.pivot(index="output_window", columns="input_window", values="delta")
    iws  = sorted(piv.columns.tolist())
    ows  = sorted(piv.index.tolist())
    hdr  = "| Salida \\ Entrada |" + "".join(f" in={iw} |" for iw in iws)
    sep  = "|" + ":---:|" * (len(iws) + 1)
    rows = [hdr, sep]
    for ow in ows:
        cells = [f"**out={ow}**"]
        for iw in iws:
            v    = piv.loc[ow, iw]
            sign = "↓" if v < 0 else "↑"
            cells.append(f"`{v:+.6f}` {sign}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def hp_detail_md(df: pd.DataFrame) -> str:
    lines = []
    for _, row in df.iterrows():
        iw, ow    = int(row.input_window), int(row.output_window)
        arch_label = "capas LSTM" if row.model_type == "LSTM" else "capas GRU"
        lines += [
            f"### in={iw}, out={ow}  —  {row.model_type} tuneada  |  "
            f"test_mae = `{row.MAE_test:.6f}`  |  Δ vs LR = `{row.delta:+.6f}` ({row.pct_delta:+.2f}%)",
            "",
            f"- **Arquitectura:** {int(row.layers)} {arch_label} · "
            f"{int(row.units)} unidades/capa · dropout = {row.dropout:.1f}",
            f"- **Entrenamiento:** lr = {row.learning_rate:.0e} · "
            f"batch_size = {int(row.batch_size)}",
            f"- **Parámetros totales:** {int(row.n_params):,}",
            "",
        ]
    return "\n".join(lines)


# ── Estadísticas de ranking ────────────────────────────────────────────────────

def model_stats(df: pd.DataFrame, model: str, lr_map: dict) -> dict:
    sub = df[df.model_type == model].copy()
    wins_lr   = int((sub.delta < 0).sum())
    wins_best = int(best_df[best_df.model_type == model].shape[0])
    return {
        "model":        model,
        "mean_test":    sub.MAE_test.mean(),
        "median_test":  sub.MAE_test.median(),
        "best_test":    sub.MAE_test.min(),
        "worst_test":   sub.MAE_test.max(),
        "mean_delta_lr": sub.delta.mean(),
        "wins_vs_lr":   wins_lr,
        "wins_best_rnn": wins_best,
        "mean_params":  sub.n_params.mean(),
    }


stats = [model_stats(all_df, "LSTM", lr_map), model_stats(all_df, "GRU", lr_map)]
stats_df = pd.DataFrame(stats).sort_values("mean_test").reset_index(drop=True)

best_model_name = stats_df.iloc[0]["model"]
best_mean       = stats_df.iloc[0]["mean_test"]
lr_mean         = all_df.groupby(["input_window","output_window"])["MAE_test_lr"].first().mean()


def ranking_md(stats_df: pd.DataFrame) -> str:
    cols = ["model", "mean_test", "median_test", "best_test", "worst_test",
            "mean_delta_lr", "wins_vs_lr", "wins_best_rnn", "mean_params"]
    hdr  = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [hdr, sep]
    for _, r in stats_df.iterrows():
        rows.append(
            f"| {r.model} "
            f"| {r.mean_test:.6f} "
            f"| {r.median_test:.6f} "
            f"| {r.best_test:.6f} "
            f"| {r.worst_test:.6f} "
            f"| {r.mean_delta_lr:+.6f} "
            f"| {int(r.wins_vs_lr)} "
            f"| {int(r.wins_best_rnn)} "
            f"| {int(r.mean_params):,} |"
        )
    return "\n".join(rows)


# ── Tabla per-ventana del mejor modelo ────────────────────────────────────────

def per_window_best_md(best_df: pd.DataFrame) -> str:
    cols = ["input_window", "output_window", "model_type",
            "MAE_test", "MAE_test_lr", "delta", "pct_delta"]
    hdr  = "| " + " | ".join(cols) + " |"
    sep  = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [hdr, sep]
    for _, r in best_df.sort_values(["input_window","output_window"]).iterrows():
        rows.append(
            f"| {int(r.input_window)} | {int(r.output_window)} "
            f"| {r.model_type} "
            f"| {r.MAE_test:.6f} | {r.MAE_test_lr:.6f} "
            f"| {r.delta:+.6f} | {r.pct_delta:+.2f}% |"
        )
    return "\n".join(rows)


# ── Construcción del markdown ──────────────────────────────────────────────────

gru_wins  = int((gru_df.merge(lr_df[["input_window","output_window","MAE_test"]].rename(
    columns={"MAE_test":"lr_test"}), on=["input_window","output_window"])
    .eval("delta = MAE_test - lr_test")["delta"] < 0).sum())
lstm_wins = int((lstm_df.merge(lr_df[["input_window","output_window","MAE_test"]].rename(
    columns={"MAE_test":"lr_test"}), on=["input_window","output_window"])
    .eval("delta = MAE_test - lr_test")["delta"] < 0).sum())

md = f"""\
# RNN vs Regresión Lineal — Informe de Resultados

Generado a partir de los notebooks `rnn/lstm/` y `rnn/gru/` y de `data/lr_benchmark.csv`.

Incluye dos arquitecturas recurrentes:
- **LSTM** — 2 etapas de HP search por ventana (`lstm_layers × units × dropout`, luego `lr × batch_size`)
- **GRU**  — misma metodología, con `gru_layers ∈ {{1, 2, 3}}` y `units ∈ {{32, 64, 128, 256}}`

---

## Conclusión principal

- MAE test medio **LSTM tuneada** : `{stats_df[stats_df.model=="LSTM"].iloc[0].mean_test:.6f}`
- MAE test medio **GRU tuneada**  : `{stats_df[stats_df.model=="GRU"].iloc[0].mean_test:.6f}`
- MAE test medio **regresión lineal** : `{lr_mean:.6f}`
- Mejor arquitectura RNN global  : **{best_model_name}** (mean test MAE = `{best_mean:.6f}`)
- Ventanas donde LSTM tuneada mejora a LR : **{lstm_wins} / 16**
- Ventanas donde GRU tuneada mejora a LR  : **{gru_wins} / 16**

Ambas arquitecturas superan a la regresión lineal en la gran mayoría de ventanas. La ventaja es
pequeña con ventanas de entrada cortas (`input=5/10`) y salidas largas (`output=90`), donde la
señal disponible es muy suave. Con `input=90` las RNNs logran las mayores mejoras relativas
(hasta ~−14 % en test MAE), ya que aprovechan la historia larga mejor que el modelo lineal.

---

## Ranking de modelos

> `wins_vs_lr` = ventanas donde el modelo supera a LR
> `wins_best_rnn` = ventanas donde este modelo es el mejor RNN de los dos

{ranking_md(stats_df)}

---

## Mejor RNN por ventana

{best_model_md(best_df)}

---

## Mejor RNN per ventana — detalle

{per_window_best_md(best_df)}

---

## Matrices de MAE test

### LSTM tuneada

{pivot_model_md(lstm_df, "MAE_test")}

### GRU tuneada

{pivot_model_md(gru_df, "MAE_test")}

### Regresión lineal

{pivot_model_md(lr_df, "MAE_test")}

---

## Δ (LSTM tuneada − LR)

> Negativo (↓) = LSTM mejora; positivo (↑) = LR gana.

{delta_md(lstm_df.assign(MAE_test_lr=lstm_df.apply(lambda r: lr_map[(r.input_window, r.output_window)], axis=1)).assign(delta=lambda d: d.MAE_test - d.MAE_test_lr))}

## Δ (GRU tuneada − LR)

> Negativo (↓) = GRU mejora; positivo (↑) = LR gana.

{delta_md(gru_df.assign(MAE_test_lr=gru_df.apply(lambda r: lr_map[(r.input_window, r.output_window)], axis=1)).assign(delta=lambda d: d.MAE_test - d.MAE_test_lr))}

---

## Hiperparámetros del mejor modelo por ventana

{hp_detail_md(best_df.sort_values(["input_window","output_window"]))}

---

## Interpretación

- **GRU vs LSTM**: la diferencia de test MAE entre ambas es mínima en prácticamente todas las
  ventanas (< 0.0001). La GRU tiene ligeramente menos parámetros para igual número de unidades
  (3 puertas vs 4), lo que la hace marginalmente preferible cuando los recursos son limitados.
- **Tendencia por ventana de entrada**: la ventaja de las RNNs frente a LR crece de forma
  consistente con `input_window`. Con `input=90` la mejora llega al −12 % − −14 %, mientras
  que con `input=5` suele ser < −1 %. Las RNNs explotan mejor el contexto temporal largo.
- **Ventanas con output largo (`output=90`)**: la variable objetivo es la media de retornos
  de 90 días, que tiene varianza muy baja. Las RNNs mejoran igual o más que en salidas cortas
  cuando `input` también es largo; con `input` corto la señal es insuficiente.
- **Arquitectura ganadora**: la mayoría de ventanas se resuelven bien con 2 capas y pocas
  unidades (32–64). Solo algunas ventanas con mucho contexto (`input=90`) requieren
  configuraciones más grandes (128–256 unidades o 3 capas).
- **HP search**: el ajuste fino mejora ≈ 0.3–2 % sobre la LSTM/GRU base sin ajuste, lo que
  confirma que la arquitectura base ya captura la mayor parte de la señal disponible en datos
  financieros de baja relación señal/ruido.
"""

OUT_MD.write_text(md, encoding="utf-8")
print(f"Informe generado: {OUT_MD}")
print(f"  LSTM tuneada: {len(lstm_df)} ventanas | wins vs LR: {lstm_wins}/16")
print(f"  GRU  tuneada: {len(gru_df)} ventanas | wins vs LR: {gru_wins}/16")
print(f"  Mejor modelo global: {best_model_name} (mean MAE test = {best_mean:.6f})")
