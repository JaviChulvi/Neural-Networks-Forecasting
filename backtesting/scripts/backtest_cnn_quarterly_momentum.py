"""Backtesting trimestral del CNN Deep Conv1D sobre el año 2025.

Estrategias
-----------
cnn_top4_equal         : Top-4 por retorno predicho; pesos iguales (25 % cada uno).
cnn_top4_riskadjusted  : Top-4 por (retorno predicho / volatilidad trailing); pesos iguales.
cnn_top4_threshold     : Como cnn_top4_equal, pero solo activos con predicción > umbral;
                         el resto va a cash (retorno 0 %).
bench_momentum         : Top-4 por retorno acumulado realizado en los 90 días anteriores.
bench_equalweight      : Buy & Hold con peso igual en los 23 activos (benchmark pasivo).

Uso como script
---------------
    python backtesting/scripts/backtest_cnn_quarterly_momentum.py
    CNN_INPUT_WINDOW=30 CNN_OUTPUT_WINDOW=90 python backtesting/scripts/backtest_cnn_quarterly_momentum.py
    CNN_RELATIVE=1 python backtesting/scripts/backtest_cnn_quarterly_momentum.py

Uso como módulo (desde un notebook, por ejemplo)
-------------------------------------------------
    import backtest_cnn_quarterly_momentum as bt
    bt.setup()                                        # input=10, output=90, absoluto
    bt.setup(input_window=30)                         # input=30, output=90, absoluto
    bt.setup(input_window=10, output_window=90, relative=True)  # target relativo
    ret_df, holdings = bt.run_backtest()
    metrics = bt.compute_metrics('cnn_top4_equal', ret_df, holdings)

Salidas (solo al ejecutar como script)
---------------------------------------
    data/backtest/portfolio_values_momentum_in{INPUT}_out{OUTPUT}.csv          (absoluto)
    data/backtest/portfolio_values_momentum_rel_in{INPUT}_out{OUTPUT}.csv       (relativo)
    data/backtest/metrics_summary_momentum[_rel]_in{INPUT}_out{OUTPUT}.csv
    data/backtest/cumulative_return_momentum[_rel]_in{INPUT}_out{OUTPUT}.png
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model" / "cnn"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Any
import keras

from util import load_returns, get_train_test
from cnn_utils import MODELS_DIR, split_train_val

# ── Configuración ─────────────────────────────────────────────────────────────
INPUT_WINDOW = int(os.getenv("CNN_INPUT_WINDOW", "10"))
OUTPUT_WINDOW = int(os.getenv("CNN_OUTPUT_WINDOW", "90"))
RELATIVE = os.getenv("CNN_RELATIVE", "0") == "1"
N_TOP = 4
TRAILING_VOL_DAYS = 90
PRED_THRESHOLD = 1e-4   # media diaria log-return mínima (~1 % en 90 días de holding)
RISK_FREE_ANNUAL = 0.04

OUTPUT_DIR = PROJECT_ROOT / "data" / "backtest"

# ── Estado del módulo (poblado por setup()) ───────────────────────────────────
# Typed as Any: estas variables se asignan en setup(); usarlas antes lanza NameError.
model: Any
scaler: Any
returns_full: Any
assets: Any
n_assets: Any
rebalance_dates: Any
period_ends: Any
end_of_backtest: Any
relative: Any


# ── Setup ─────────────────────────────────────────────────────────────────────
def setup(
    input_window: int | None = None,
    output_window: int | None = None,
    relative: bool | None = None,
) -> None:
    """Carga el modelo, ajusta el scaler y configura los periodos de rebalanceo.

    Parameters
    ----------
    input_window  : Tamaño de la ventana de entrada (días). Si se omite, usa
                    CNN_INPUT_WINDOW del entorno o el valor por defecto (10).
    output_window : Tamaño de la ventana de salida (días). Si se omite, usa
                    CNN_OUTPUT_WINDOW del entorno o el valor por defecto (90).
    relative      : Si True, carga el modelo entrenado con target relativo al
                    universo (_rel). Si se omite, usa CNN_RELATIVE del entorno
                    o False por defecto.

    Debe llamarse antes de usar cualquier otra función del módulo.
    Puede llamarse de nuevo con distintos parámetros para cambiar de modelo.
    """
    global model, scaler, returns_full, assets, n_assets
    global rebalance_dates, period_ends, end_of_backtest
    global INPUT_WINDOW, OUTPUT_WINDOW, RELATIVE

    if input_window is not None:
        INPUT_WINDOW = input_window
    if output_window is not None:
        OUTPUT_WINDOW = output_window
    if relative is not None:
        RELATIVE = relative

    # Modelo
    suffix = "_rel" if RELATIVE else ""
    model_path = MODELS_DIR / f"cnn_input{INPUT_WINDOW}_output{OUTPUT_WINDOW}{suffix}_model.keras"
    train_script = f"train_cnn_in{INPUT_WINDOW}_out{OUTPUT_WINDOW}{'_rel' if RELATIVE else ''}_to2024.py"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}\n"
            "Entrena primero con:\n"
            f"  python backtesting/scripts/{train_script}"
        )
    model = keras.models.load_model(model_path)
    print(f"Modelo cargado: {model_path.relative_to(PROJECT_ROOT)}")

    # Scaler — replicamos exactamente el pipeline de train_cnn_window:
    #   get_train_test (90/10) → split_train_val (retira 10 % final como val)
    d = get_train_test(
        input_window_size=INPUT_WINDOW,
        output_window_size=OUTPUT_WINDOW,
        returns_file="returns_to2024.parquet",
        relative=RELATIVE,
    )
    X_train_final, _, _, _ = split_train_val(d.X_train, d.y_train)
    n, w, a = X_train_final.shape
    scaler = StandardScaler()
    scaler.fit(X_train_final.reshape(n, w * a))
    print(f"Scaler ajustado sobre {n} secuencias ({w}d × {a} activos).")

    # Datos completos (incluye 2025)
    returns_full = load_returns(filename="returns.parquet")
    assets = list(returns_full.columns)
    n_assets = len(assets)

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
    iloc = returns_full.index.get_loc(date)
    window = returns_full.iloc[iloc - INPUT_WINDOW : iloc].values
    X = window.reshape(1, INPUT_WINDOW, n_assets)
    X_scaled = scaler.transform(X.reshape(1, -1)).reshape(1, INPUT_WINDOW, n_assets)
    return model.predict(X_scaled, verbose=0)[0]


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
def strat_cnn_equal(pred: np.ndarray, **_) -> dict:
    top = np.argsort(pred)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_cnn_riskadjusted(pred: np.ndarray, vol: np.ndarray, **_) -> dict:
    signal = pred / vol
    top = np.argsort(signal)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_cnn_threshold(pred: np.ndarray, **_) -> dict:
    top = np.argsort(pred)[::-1][:N_TOP]
    selected = [i for i in top if pred[i] > PRED_THRESHOLD]
    if not selected:
        return {}  # 100 % cash
    return {assets[i]: 1.0 / len(selected) for i in selected}


def strat_momentum(date: pd.Timestamp, **_) -> dict:
    iloc = returns_full.index.get_loc(date)
    window = returns_full.iloc[max(0, iloc - TRAILING_VOL_DAYS) : iloc]
    cum = window.sum(axis=0)
    top = np.argsort(cum.values)[::-1][:N_TOP]
    return {assets[i]: 1.0 / N_TOP for i in top}


def strat_equalweight(**_) -> dict:
    return {a: 1.0 / n_assets for a in assets}


STRATEGIES = {
    "cnn_top4_equal":        strat_cnn_equal,
    "cnn_top4_riskadjusted": strat_cnn_riskadjusted,
    "cnn_top4_threshold":    strat_cnn_threshold,
    "bench_momentum":        strat_momentum,
    "bench_equalweight":     strat_equalweight,
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
    rel_part = "_rel" if RELATIVE else ""
    tag = f"momentum{rel_part}_in{INPUT_WINDOW}_out{OUTPUT_WINDOW}"

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
        "cnn_top4_equal":        dict(color="royalblue",  lw=2,   ls="-",  label="CNN Top-4 Equal"),
        "cnn_top4_riskadjusted": dict(color="darkorange", lw=2,   ls="-",  label="CNN Top-4 Risk-Adj."),
        "cnn_top4_threshold":    dict(color="seagreen",   lw=2,   ls="-",  label="CNN Top-4 + Umbral"),
        "bench_momentum":        dict(color="purple",     lw=1.5, ls="-.", label="Momentum"),
        "bench_equalweight":     dict(color="gray",       lw=1.5, ls="--", label="Equal Weight (B&H)"),
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, style in STYLE.items():
        ax.plot(cum_value.index, cum_value[name], **style)
    for rb in rebalance_dates[1:]:
        ax.axvline(rb, color="black", lw=0.7, ls=":", alpha=0.5)
    ax.axhline(100, color="black", lw=0.8, ls=":")
    target_label = "target relativo" if RELATIVE else "target absoluto"
    ax.set_title(
        f"Backtesting trimestral 2025 — CNN Deep Conv1D "
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
