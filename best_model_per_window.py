"""
Compare MLP, CNN, RNN, and Mixtos families and find the global best model per
(input_window, output_window) combination by test MAE.
Writes best_model_per_window_report.md to the project root.

Data extracted from:
  model/mlp/mlp_vs_lr_report.md
  model/cnn/cnn_vs_lr_report.md
  model/rnn/rnn_vs_lr_report.md
  model/mixtos/mixtos_vs_lr_report.md
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# LR benchmark — shared across all families
# ---------------------------------------------------------------------------
LR = {
    (5,  1): 0.012384, (5,  5): 0.005625, (5,  30): 0.002340, (5,  90): 0.001271,
    (10, 1): 0.012554, (10, 5): 0.005698, (10, 30): 0.002358, (10, 90): 0.001282,
    (30, 1): 0.012924, (30, 5): 0.005877, (30, 30): 0.002436, (30, 90): 0.001351,
    (90, 1): 0.014095, (90, 5): 0.006348, (90, 30): 0.002628, (90, 90): 0.001518,
}

# ---------------------------------------------------------------------------
# Best model per window per family (lowest test MAE from each family report)
# ---------------------------------------------------------------------------

MLP_BEST = {
    (5,  1):  {"model": "mlp_3x128_gelu_dropout_l2", "mae": 0.012224},
    (5,  5):  {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.005574},
    (5,  30): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.002321},
    (5,  90): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.001266},
    (10, 1):  {"model": "mlp_3x128_gelu_dropout_l2", "mae": 0.012225},
    (10, 5):  {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.005573},
    (10, 30): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.002321},
    (10, 90): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.001263},
    (30, 1):  {"model": "mlp_3x128_gelu_dropout_l2", "mae": 0.012232},
    (30, 5):  {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.005574},
    (30, 30): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.002323},
    (30, 90): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.001264},
    (90, 1):  {"model": "mlp_3x128_gelu_dropout_l2", "mae": 0.012249},
    (90, 5):  {"model": "mlp_3x128_gelu_dropout_l2", "mae": 0.005605},
    (90, 30): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.002323},
    (90, 90): {"model": "mlp_4x100_gelu_dropout_l2", "mae": 0.001268},
}

CNN_BEST = {
    (5,  1):  {"model": "CNN_Deep_Conv1D", "mae": 0.012237},
    (5,  5):  {"model": "CNN_Deep_Conv1D", "mae": 0.0055824},
    (5,  30): {"model": "CNN_Deep_Conv1D", "mae": 0.00232186},
    (5,  90): {"model": "CNN_Deep_Conv1D", "mae": 0.0012622},
    (10, 1):  {"model": "CNN_Deep_Conv1D", "mae": 0.0122385},
    (10, 5):  {"model": "CNN_Deep_Conv1D", "mae": 0.00557466},
    (10, 30): {"model": "CNN_Deep_Conv1D", "mae": 0.00232092},
    (10, 90): {"model": "CNN_Deep_Conv1D", "mae": 0.00125936},
    (30, 1):  {"model": "CNN_Deep_Conv1D", "mae": 0.0122434},
    (30, 5):  {"model": "CNN_Deep_Conv1D", "mae": 0.0055767},
    (30, 30): {"model": "CNN_Deep_Conv1D", "mae": 0.00231923},
    (30, 90): {"model": "CNN_Deep_Conv1D", "mae": 0.0012626},
    (90, 1):  {"model": "CNN_Deep_Conv1D", "mae": 0.0122595},
    (90, 5):  {"model": "CNN_Deep_Conv1D", "mae": 0.00558627},
    (90, 30): {"model": "CNN_Deep_Conv1D", "mae": 0.00232253},
    (90, 90): {"model": "CNN_Deep_Conv1D", "mae": 0.00126382},
}

RNN_BEST = {
    (5,  1):  {"model": "LSTM", "mae": 0.012238},
    (5,  5):  {"model": "LSTM", "mae": 0.005586},
    (5,  30): {"model": "LSTM", "mae": 0.002325},
    (5,  90): {"model": "LSTM", "mae": 0.001275},
    (10, 1):  {"model": "LSTM", "mae": 0.012234},
    (10, 5):  {"model": "LSTM", "mae": 0.005582},
    (10, 30): {"model": "LSTM", "mae": 0.002334},
    (10, 90): {"model": "GRU",  "mae": 0.001270},
    (30, 1):  {"model": "GRU",  "mae": 0.012240},
    (30, 5):  {"model": "LSTM", "mae": 0.005584},
    (30, 30): {"model": "LSTM", "mae": 0.002340},
    (30, 90): {"model": "GRU",  "mae": 0.001269},
    (90, 1):  {"model": "LSTM", "mae": 0.012256},
    (90, 5):  {"model": "LSTM", "mae": 0.005594},
    (90, 30): {"model": "LSTM", "mae": 0.002343},
    (90, 90): {"model": "GRU",  "mae": 0.001288},
}

MIXTOS_BEST = {
    (5,  1):  {"model": "LSTM (tuned)",     "mae": 0.012232},
    (5,  5):  {"model": "GRU (tuned)",      "mae": 0.005593},
    (5,  30): {"model": "CNN-LSTM (tuned)", "mae": 0.002339},
    (5,  90): {"model": "LSTM (tuned)",     "mae": 0.001275},
    (10, 1):  {"model": "CNN-LSTM (tuned)", "mae": 0.012244},
    (10, 5):  {"model": "CNN-GRU (tuned)",  "mae": 0.005620},
    (10, 30): {"model": "LSTM (tuned)",     "mae": 0.002334},
    (10, 90): {"model": "CNN-GRU (tuned)",  "mae": 0.001272},
    (30, 1):  {"model": "Hybrid CNN-LSTM",  "mae": 0.012244},
    (30, 5):  {"model": "Hybrid CNN-GRU",   "mae": 0.005581},
    (30, 30): {"model": "LSTM (tuned)",     "mae": 0.002340},
    (30, 90): {"model": "CNN-LSTM (tuned)", "mae": 0.001346},
    (90, 1):  {"model": "CNN-LSTM (tuned)", "mae": 0.012279},
    (90, 5):  {"model": "LSTM (tuned)",     "mae": 0.005594},
    (90, 30): {"model": "LSTM (tuned)",     "mae": 0.002344},
    (90, 90): {"model": "CNN-GRU (tuned)",  "mae": 0.001396},
}

FAMILIES = {
    "MLP":    MLP_BEST,
    "CNN":    CNN_BEST,
    "RNN":    RNN_BEST,
    "Mixtos": MIXTOS_BEST,
}

INPUT_WINDOWS  = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]
WINDOWS = [(i, o) for i in INPUT_WINDOWS for o in OUTPUT_WINDOWS]


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

    # -- Interpretation --
    lines += [
        "## Interpretación",
        "",
        "- **Empate CNN–MLP (8–8)**: ambas familias se reparten las 16 ventanas a partes iguales.",
        "  RNN y Mixtos no ganan ninguna ventana cuando se comparan contra las cuatro familias.",
        "- **Patrón por output_window**: MLP domina las salidas cortas (`out=1` y `out=5`),"
        "  mientras que CNN gana la mayoría de las salidas largas (`out=30` y `out=90`).",
        "  La única excepción relevante es `in=90, out=1` donde MLP aún gana.",
        "- **Patrón por input_window**: con `input=5`, MLP gana 3 de 4 ventanas de salida;",
        "  con `input=90`, CNN gana 3 de 4. Para ventanas intermedias el reparto es mixto.",
        "- **Márgenes muy pequeños**: en la mayoría de ventanas la diferencia CNN–MLP es",
        "  < 0.0001 MAE, dentro de la variabilidad aleatoria del entrenamiento. En la práctica",
        "  ambas arquitecturas capturan la misma señal de esta serie financiera de bajo SNR.",
        "- **RNN y Mixtos**: la complejidad recurrente añadida (LSTM/GRU puro o CNN-RNN híbrido)",
        "  no reporta mejora sistemática sobre MLP o CNN en ninguna ventana.",
        "- **16/16 ventanas superan a LR**: el modelo global óptimo mejora a la regresión lineal",
        "  en las 16 combinaciones, incluyendo `(in=5, out=90)` donde los modelos recurrentes",
        "  no conseguían superar el benchmark (CNN sí lo hace con −0.72%).",
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
