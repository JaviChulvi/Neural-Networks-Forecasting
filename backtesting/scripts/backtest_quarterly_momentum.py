"""Backtesting trimestral CNN / LSTM / GRU sobre el año 2025.

El modelo a usar se elige con la variable de entorno BACKTEST_MODEL
(valores: "cnn", "lstm" o "gru"; por defecto "cnn").

Estrategias
-----------
model_top4_equal         : Top-4 por retorno predicho; pesos iguales (25 % cada uno).
model_top4_riskadjusted  : Top-4 por (retorno predicho / volatilidad trailing); pesos iguales.
model_top4_threshold     : Como model_top4_equal, pero solo activos con predicción > umbral;
                           el resto va a cash (retorno 0 %).
model_top4_propweights   : Top-4 por retorno predicho; pesos proporcionales a la predicción
                           (solo positivos; si ninguno es positivo, 100 % cash).
model_longshort          : Long top-4 / short bottom-4 por retorno predicho; pesos iguales
                           en cada pata (±1/N_TOP). Estrategia market-neutral.
bench_momentum           : Top-4 por retorno acumulado realizado en los 90 días anteriores.
bench_momentum_input     : Top-4 por retorno acumulado en los últimos INPUT_WINDOW días
                           (mismo lookback que el modelo; comparación equitativa en información).
bench_equalweight        : Buy & Hold con peso igual en los 23 activos (benchmark pasivo).

Uso como script
---------------
    python backtesting/scripts/backtest_quarterly_momentum.py
    BACKTEST_MODEL=lstm python backtesting/scripts/backtest_quarterly_momentum.py
    BACKTEST_MODEL=gru  python backtesting/scripts/backtest_quarterly_momentum.py
    BACKTEST_MODEL=cnn CNN_INPUT_WINDOW=30 python backtesting/scripts/backtest_quarterly_momentum.py
    CNN_RELATIVE=1 python backtesting/scripts/backtest_quarterly_momentum.py  # solo CNN

Uso como módulo (desde un notebook, por ejemplo)
-------------------------------------------------
    import backtest_quarterly_momentum as bt
    bt.setup()                                              # CNN input=10, output=90, absoluto
    bt.setup(model_type="lstm")                             # LSTM input=10, output=90
    bt.setup(model_type="gru")                              # GRU input=10, output=90
    bt.setup(model_type="cnn", input_window=30)             # CNN input=30, output=90
    bt.setup(model_type="cnn", relative=True)               # CNN target relativo
    ret_df, holdings = bt.run_backtest()
    metrics = bt.compute_metrics('model_top4_equal', ret_df, holdings)

Salidas (solo al ejecutar como script)
---------------------------------------
    data/backtest/portfolio_values_momentum_{model}[_rel]_in{INPUT}_out{OUTPUT}.csv
    data/backtest/metrics_summary_momentum_{model}[_rel]_in{INPUT}_out{OUTPUT}.csv
    data/backtest/cumulative_return_momentum_{model}[_rel]_in{INPUT}_out{OUTPUT}.png
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model" / "cnn"))  # necesario para cnn_utils (CNN)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Any
import keras

from util import load_returns, load_precios_close, get_train_test, build_ffd_series

# ── Configuración ─────────────────────────────────────────────────────────────
MODEL_TYPE    = os.getenv("BACKTEST_MODEL",    "cnn")   # "cnn", "lstm" o "gru"
INPUT_WINDOW  = int(os.getenv("CNN_INPUT_WINDOW",  "10"))
OUTPUT_WINDOW = int(os.getenv("CNN_OUTPUT_WINDOW", "90"))
RELATIVE      = os.getenv("CNN_RELATIVE", "0") == "1"
N_TOP = 4
TRAILING_VOL_DAYS = 90
PRED_THRESHOLD = 1e-4   # media diaria log-return mínima (~1 % en 90 días de holding)
RISK_FREE_ANNUAL = 0.04

RNN_MODELS_DIR = PROJECT_ROOT / "data" / "rnn" / "saved_models"
OUTPUT_DIR     = PROJECT_ROOT / "data" / "backtest"

# ── Estado del módulo (poblado por setup()) ───────────────────────────────────
# Typed as Any: estas variables se asignan en setup(); usarlas antes lanza NameError.
model: Any
scaler: Any          # StandardScaler para CNN; None para LSTM/GRU
returns_full: Any
assets: Any
n_assets: Any
rebalance_dates: Any
period_ends: Any
end_of_backtest: Any
relative: Any
BAR_TYPE: Any        # str | None — tipo de bars usado; None = comportamiento original
PREPROCESSING_DIR: Any  # Path | None — ruta base de los NPZ
FFD: Any             # bool — si True aplica diferenciación fraccionaria
ffd_series: Any      # pd.DataFrame | None — serie FFD para inferencia (solo cuando FFD=True)


def _resolve_model_path(mtype: str, in_w: int, out_w: int, rel: bool, bar_type: str | None = None) -> Path:
    bar_suffix = f"_{bar_type}" if bar_type is not None else ""
    if mtype == "cnn":
        from cnn_utils import MODELS_DIR as CNN_MODELS_DIR
        rel_suffix = "_rel" if rel else ""
        return CNN_MODELS_DIR / f"cnn_input{in_w}_output{out_w}{bar_suffix}{rel_suffix}_model.keras"
    # lstm o gru — misma convención de nombres
    return RNN_MODELS_DIR / f"rnn_{mtype}_input{in_w}_output{out_w}{bar_suffix}_model.keras"


# Mapeo (model_type, input_window, output_window, relative) → nombre de módulo de entrenamiento.
# Añade aquí nuevas combinaciones cuando se cree el script correspondiente.
TRAIN_SCRIPTS: dict[tuple[str, int, int, bool], str] = {
    ("cnn",  10, 90, False): "train_cnn_in10_out90_to2024",
    ("cnn",  10, 90, True):  "train_cnn_in10_out90_rel_to2024",
    ("cnn",  30, 90, False): "train_cnn_in30_out90_to2024",
    ("lstm", 10, 90, False): "train_lstm_in10_out90_to2024",
    ("lstm", 30, 90, False): "train_lstm_in30_out90_to2024",
    ("gru",  10, 90, False): "train_gru_in10_out90_to2024",
}


def _auto_train(
    mtype: str,
    in_w: int,
    out_w: int,
    rel: bool,
    bar_type: str | None = None,
    preprocessing_dir: Path | None = None,
    ffd: bool = False,
) -> None:
    """Importa y llama run() del script de entrenamiento correspondiente."""
    import importlib

    key = (mtype, in_w, out_w, rel)
    module_name = TRAIN_SCRIPTS.get(key)
    if module_name is None:
        raise FileNotFoundError(
            f"Modelo no encontrado y no hay script de entrenamiento registrado para "
            f"model={mtype!r}, input={in_w}, output={out_w}, relative={rel}.\n"
            "Entrena el modelo manualmente y vuelve a llamar setup()."
        )

    scripts_dir = str(PROJECT_ROOT / "backtesting" / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    print(f"Modelo no encontrado. Entrenando con {module_name}.run() ...")
    train_module = importlib.import_module(module_name)
    train_module.run(bar_type=bar_type, preprocessing_dir=preprocessing_dir, ffd=ffd)


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup(
    input_window: int | None = None,
    output_window: int | None = None,
    relative: bool | None = None,
    model_type: str | None = None,
    bar_type: str | None = None,
    preprocessing_dir: Path | None = None,
    ffd: bool = False,
) -> None:
    """Carga el modelo, ajusta el scaler (solo CNN) y configura los periodos de rebalanceo.

    Parameters
    ----------
    input_window       : Tamaño de la ventana de entrada (días). Por defecto CNN_INPUT_WINDOW / 10.
    output_window      : Tamaño de la ventana de salida (días). Por defecto CNN_OUTPUT_WINDOW / 90.
    relative           : Solo aplicable al modelo CNN. Si True, carga la variante _rel.
                         Se ignora para LSTM y GRU (no existe variante relativa).
    model_type         : "cnn", "lstm" o "gru". Por defecto BACKTEST_MODEL / "cnn".
    bar_type           : "time", "count", "volume" o "dollar". Si se indica, carga los NPZ
                         pre-generados por 03_build_preprocessed_sequences.py en lugar del
                         parquet original. None usa el comportamiento original.
    preprocessing_dir  : Ruta base de los NPZ. Solo relevante si bar_type no es None.
    ffd                : Si True, aplica diferenciación fraccionaria sobre las series antes
                         del entrenamiento y del ajuste del scaler.

    Debe llamarse antes de usar cualquier otra función del módulo.
    Puede llamarse de nuevo con distintos parámetros para cambiar de modelo.
    """
    global model, scaler, returns_full, assets, n_assets
    global rebalance_dates, period_ends, end_of_backtest
    global INPUT_WINDOW, OUTPUT_WINDOW, RELATIVE, MODEL_TYPE, BAR_TYPE, PREPROCESSING_DIR, FFD, ffd_series

    if input_window is not None:
        INPUT_WINDOW = input_window
    if output_window is not None:
        OUTPUT_WINDOW = output_window
    if relative is not None:
        RELATIVE = relative
    if model_type is not None:
        MODEL_TYPE = model_type
    BAR_TYPE = bar_type
    PREPROCESSING_DIR = preprocessing_dir
    FFD = ffd

    if MODEL_TYPE not in ("cnn", "lstm", "gru"):
        raise ValueError(
            f"model_type debe ser 'cnn', 'lstm' o 'gru', recibido: {MODEL_TYPE!r}"
        )
    if MODEL_TYPE in ("lstm", "gru") and RELATIVE:
        raise ValueError(
            f"El modelo {MODEL_TYPE.upper()} no tiene variante relativa (_rel)."
        )

    # Modelo — se reentrena siempre para garantizar que solo usa datos previos a 2025
    path = _resolve_model_path(MODEL_TYPE, INPUT_WINDOW, OUTPUT_WINDOW, RELATIVE, BAR_TYPE)
    _auto_train(MODEL_TYPE, INPUT_WINDOW, OUTPUT_WINDOW, RELATIVE, BAR_TYPE, PREPROCESSING_DIR, FFD)
    model = keras.models.load_model(path)
    print(f"Modelo cargado: {path.relative_to(PROJECT_ROOT)}")

    # Scaler — solo CNN (LSTM y GRU fueron entrenados sobre retornos crudos sin escalado)
    if MODEL_TYPE == "cnn":
        from cnn_utils import split_train_val
        d = get_train_test(
            input_window_size=INPUT_WINDOW,
            output_window_size=OUTPUT_WINDOW,
            returns_file="returns_to2024.parquet",
            relative=RELATIVE,
            bar_type=BAR_TYPE,
            preprocessing_dir=PREPROCESSING_DIR,
            ffd=FFD,
        )
        X_train_final, _, _, _ = split_train_val(d.X_train, d.y_train)
        n, w, a = X_train_final.shape
        scaler = StandardScaler()
        scaler.fit(X_train_final.reshape(n, w * a))
        print(f"Scaler ajustado sobre {n} secuencias ({w}d × {a} activos).")
    else:
        scaler = None
        print(f"{MODEL_TYPE.upper()}: sin escalado de inputs.")

    # Datos completos (incluye 2025)
    returns_full = load_returns(filename="returns.parquet")
    assets = list(returns_full.columns)
    n_assets = len(assets)

    # Serie FFD para inferencia: misma transformación que en entrenamiento
    if FFD:
        _eff_bar = BAR_TYPE if BAR_TYPE is not None else "raw"
        if _eff_bar == "raw":
            close_df = load_precios_close()
        else:
            close_df = pd.read_parquet(
                PROJECT_ROOT / "data" / "preprocessing" / f"{_eff_bar}_bars_close.parquet"
            )
        d_csv = PROJECT_ROOT / "data" / "preprocessing" / f"{_eff_bar}_bars_ffd_d_values.csv"
        ffd_series = build_ffd_series(close_df, d_csv).reindex(columns=assets)
        print(
            f"FFD series: {ffd_series.shape}  "
            f"({ffd_series.index.min().date()} → {ffd_series.index.max().date()})"
        )
    else:
        ffd_series = None

    dates_2025 = returns_full.index[returns_full.index.year == 2025]
    print(
        f"Universo: {n_assets} activos | "
        f"2025: {dates_2025[0].date()} → {dates_2025[-1].date()} ({len(dates_2025)} días)"
    )

    # Fechas de rebalanceo trimestrales
    def _first_trading_day(year: int, month: int):
        mask = (returns_full.index.year == year) & (returns_full.index.month == month)
        dates = returns_full.index[mask]
        return dates[0] if not dates.empty else None

    rebalance_dates = [
        d for d in (
            _first_trading_day(2025, 1),
            _first_trading_day(2025, 4),
            _first_trading_day(2025, 7),
            _first_trading_day(2025, 10),
        )
        if d is not None
    ]
    end_of_backtest = dates_2025[-1]
    period_ends = [rebalance_dates[i + 1] for i in range(len(rebalance_dates) - 1)] + [
        end_of_backtest + pd.Timedelta(days=1)
    ]

    print(f"Periodos de rebalanceo ({len(rebalance_dates)}):")
    for i, (rb, end) in enumerate(zip(rebalance_dates, period_ends)):
        last = returns_full.index[(returns_full.index >= rb) & (returns_full.index < end)][-1]
        print(f"  Q{i + 1}: {rb.date()} → {last.date()}")


# ── Funciones auxiliares ──────────────────────────────────────────────────────
def predict_at(date: pd.Timestamp) -> np.ndarray:
    """Predice la media de log-returns de los próximos OUTPUT_WINDOW días."""
    source = ffd_series if FFD else returns_full
    iloc = source.index.get_loc(date)
    window = source.iloc[iloc - INPUT_WINDOW : iloc].values
    X = window.reshape(1, INPUT_WINDOW, n_assets)
    if scaler is not None:
        X = scaler.transform(X.reshape(1, -1)).reshape(1, INPUT_WINDOW, n_assets)
    return model.predict(X, verbose=0)[0]


def trailing_vol(date: pd.Timestamp) -> np.ndarray:
    """Std de log-returns en los últimos TRAILING_VOL_DAYS días anteriores a date."""
    iloc = returns_full.index.get_loc(date)
    window = returns_full.iloc[max(0, iloc - TRAILING_VOL_DAYS) : iloc].values
    return window.std(axis=0) + 1e-8


def period_returns(start: pd.Timestamp, end_excl: pd.Timestamp) -> pd.DataFrame:
    mask = (returns_full.index >= start) & (returns_full.index < end_excl)
    return returns_full.loc[mask]


def portfolio_log_returns(weights: dict, ret: pd.DataFrame) -> pd.Series:
    """Retorno log diario del portfolio (aproximación lineal para retornos pequeños)."""
    w = pd.Series(0.0, index=assets)
    for k, v in weights.items():
        w[k] = v
    return ret[assets].dot(w)


# ── Estrategias ───────────────────────────────────────────────────────────────
def strat_model_equal(pred: np.ndarray, **_) -> dict:
    top = np.argsort(pred)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_model_riskadjusted(pred: np.ndarray, vol: np.ndarray, **_) -> dict:
    signal = pred / vol
    top = np.argsort(signal)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_model_threshold(pred: np.ndarray, **_) -> dict:
    top = np.argsort(pred)[::-1][:N_TOP]
    selected = [i for i in top if pred[i] > PRED_THRESHOLD]
    if not selected:
        return {}  # 100 % cash
    return {assets[i]: 1.0 / len(selected) for i in selected}


def strat_model_propweights(pred: np.ndarray, **_) -> dict:
    """Top-4 por retorno predicho; pesos proporcionales a la predicción (solo positivos)."""
    top = np.argsort(pred)[::-1][:N_TOP]
    selected = [i for i in top if pred[i] > 0]
    if not selected:
        return {}  # 100 % cash
    pos_preds = np.array([pred[i] for i in selected])
    weights = pos_preds / pos_preds.sum()
    return {assets[i]: float(w) for i, w in zip(selected, weights)}


def strat_model_longshort(pred: np.ndarray, **_) -> dict:
    """Long top-4 / short bottom-4; pesos iguales en cada pata (±1/N_TOP)."""
    order = np.argsort(pred)[::-1]
    weights = {}
    for i in order[:N_TOP]:
        weights[assets[i]] = 1.0 / N_TOP
    for i in order[-N_TOP:]:
        weights[assets[i]] = -1.0 / N_TOP
    return weights


def strat_momentum(date: pd.Timestamp, **_) -> dict:
    iloc = returns_full.index.get_loc(date)
    window = returns_full.iloc[max(0, iloc - TRAILING_VOL_DAYS) : iloc]
    cum = window.sum(axis=0)
    top = np.argsort(cum.values)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_momentum_input(date: pd.Timestamp, **_) -> dict:
    """Top-4 por retorno acumulado en los últimos INPUT_WINDOW días (mismo lookback que el modelo)."""
    iloc = returns_full.index.get_loc(date)
    window = returns_full.iloc[max(0, iloc - INPUT_WINDOW) : iloc]
    cum = window.sum(axis=0)
    top = np.argsort(cum.values)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_equalweight(**_) -> dict:
    return {a: 1.0 / n_assets for a in assets}


STRATEGIES = {
    "model_top4_equal":        strat_model_equal,
    "model_top4_riskadjusted": strat_model_riskadjusted,
    "model_top4_threshold":    strat_model_threshold,
    "model_top4_propweights":  strat_model_propweights,
    "model_longshort":         strat_model_longshort,
    "bench_momentum":          strat_momentum,
    "bench_momentum_input":    strat_momentum_input,
    "bench_equalweight":       strat_equalweight,
}


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest() -> tuple[pd.DataFrame, dict]:
    """Ejecuta el backtest trimestral para todas las estrategias.

    Returns
    -------
    ret_df   : DataFrame con retornos log diarios por estrategia
    holdings : dict {strategy_name: [set_of_assets_per_quarter]}
    """
    all_dates: list = []
    daily_log_rets: dict = {name: [] for name in STRATEGIES}
    holdings: dict = {name: [] for name in STRATEGIES}

    for q_idx, (rb_date, end_excl) in enumerate(zip(rebalance_dates, period_ends)):
        ret = period_returns(rb_date, end_excl)
        if ret.empty:
            continue

        pred = predict_at(rb_date)
        vol = trailing_vol(rb_date)
        all_dates.extend(ret.index.tolist())

        top4_names = [assets[i] for i in np.argsort(pred)[::-1][:N_TOP]]
        print(f"Q{q_idx + 1} ({rb_date.date()}): pred top-4 = {top4_names}")

        for name, fn in STRATEGIES.items():
            w = fn(pred=pred, vol=vol, date=rb_date)
            holdings[name].append(set(w.keys()))
            daily_lr = (
                portfolio_log_returns(w, ret) if w else pd.Series(0.0, index=ret.index)
            )
            daily_log_rets[name].extend(daily_lr.tolist())

    return pd.DataFrame(daily_log_rets, index=all_dates), holdings


# ── Métricas ──────────────────────────────────────────────────────────────────
def compute_metrics(name: str, ret_df: pd.DataFrame, holdings: dict) -> dict:
    """Calcula métricas de rendimiento para una estrategia."""
    lr = ret_df[name]
    rf_daily = RISK_FREE_ANNUAL / 252
    total_ret = float(np.exp(lr.sum()) - 1)
    sharpe = float((lr - rf_daily).mean() / (lr.std() + 1e-10) * np.sqrt(252))
    cum = np.exp(lr.cumsum())
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())

    asset_sets = holdings[name]
    if len(asset_sets) > 1:
        turnover = float(np.mean([
            len(asset_sets[i] ^ asset_sets[i + 1]) / max(N_TOP, 1)
            for i in range(len(asset_sets) - 1)
        ]))
    else:
        turnover = 0.0

    return {
        "strategy": name,
        "total_return_%": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_%": round(max_dd * 100, 2),
        "avg_turnover": round(turnover, 3),
    }


# ── Save outputs ──────────────────────────────────────────────────────────────
def save_outputs(ret_df: pd.DataFrame, holdings: dict) -> None:
    """Guarda CSV de valores, CSV de métricas y gráfico de retorno acumulado."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rel_part = "_rel" if (RELATIVE and MODEL_TYPE == "cnn") else ""
    bar_part = f"_{BAR_TYPE}" if BAR_TYPE is not None else ""
    tag = f"momentum_{MODEL_TYPE}{bar_part}{rel_part}_in{INPUT_WINDOW}_out{OUTPUT_WINDOW}"
    model_label = MODEL_TYPE.upper() + (f" ({BAR_TYPE})" if BAR_TYPE else "") + (" (rel)" if rel_part else "")

    cum_value = (np.exp(ret_df.cumsum()) * 100).round(4)
    values_path = OUTPUT_DIR / f"portfolio_values_{tag}.csv"
    cum_value.to_csv(values_path)
    print(f"Portfolio values → {values_path}")

    metrics = pd.DataFrame(
        [compute_metrics(n, ret_df, holdings) for n in STRATEGIES]
    ).set_index("strategy")
    metrics_path = OUTPUT_DIR / f"metrics_summary_{tag}.csv"
    metrics.to_csv(metrics_path)
    print("\nMétricas:")
    print(metrics.to_string())
    print(f"\nMétricas → {metrics_path}")

    STYLE = {
        "model_top4_equal":        dict(color="royalblue",  lw=2,   ls="-",  label=f"{model_label} Top-4 Equal"),
        "model_top4_riskadjusted": dict(color="darkorange", lw=2,   ls="-",  label=f"{model_label} Top-4 Risk-Adj."),
        "model_top4_threshold":    dict(color="seagreen",   lw=2,   ls="-",  label=f"{model_label} Top-4 + Umbral"),
        "model_top4_propweights":  dict(color="crimson",    lw=2,   ls="-",  label=f"{model_label} Top-4 Prop. Pesos"),
        "model_longshort":         dict(color="black",      lw=2,   ls="-",  label=f"{model_label} Long-Short"),
        "bench_momentum":          dict(color="purple",     lw=1.5, ls="-.", label="Momentum 90d"),
        "bench_momentum_input":    dict(color="darkgreen",  lw=1.5, ls=":",  label=f"Momentum {INPUT_WINDOW}d"),
        "bench_equalweight":       dict(color="gray",       lw=1.5, ls="--", label="Equal Weight (B&H)"),
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, style in STYLE.items():
        ax.plot(cum_value.index, cum_value[name], **style)
    for rb in rebalance_dates[1:]:
        ax.axvline(rb, color="black", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(100, color="black", lw=0.8, ls=":")
    target_label = "target relativo" if RELATIVE else "target absoluto"
    ax.set_title(
        f"Backtesting trimestral 2025 — {model_label} "
        f"(input={INPUT_WINDOW}, output={OUTPUT_WINDOW}, {target_label})",
        fontsize=13,
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor cartera (base 100)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = OUTPUT_DIR / f"cumulative_return_{tag}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Gráfico → {plot_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    setup(relative=RELATIVE)
    ret_df, holdings = run_backtest()
    save_outputs(ret_df, holdings)
