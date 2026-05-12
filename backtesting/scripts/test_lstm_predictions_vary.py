"""Verifica que las predicciones del LSTM (in=10, out=90) varían entre instantes
temporales suficientemente espaciados.

Diagrama de causalidad que el test explora:
    inputs distintos  →  ¿predicciones distintas?
                ↑                   ↑
         test_inputs_*        test_predictions_*

Si los tests de inputs pasan pero los de predicciones fallan, el modelo es un
predictor constante (aprende la mediana del target y la ignora el input).

Nota: el LSTM fue entrenado sobre retornos crudos (sin StandardScaler), a
diferencia del CNN que sí aplica escalado. Los inputs se pasan tal cual.

Ejecución:
    pytest backtesting/scripts/test_lstm_predictions_vary.py -v
    python backtesting/scripts/test_lstm_predictions_vary.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import keras  # noqa: E402
from util import DATA_DIR, create_time_series_data, load_returns  # noqa: E402

# ── parámetros del modelo ──────────────────────────────────────────────────
INPUT_WINDOW = 10
OUTPUT_WINDOW = 90
N_SAMPLES = 100
MIN_STEP = INPUT_WINDOW        # garantiza que ningún par comparte días de entrada
MODEL_PATH = PROJECT_ROOT / "data" / "rnn" / "saved_models" / "rnn_lstm_input10_output90_model.keras"
RETURNS_FILE = "returns_to2024.parquet"


# ── fixture compartida entre tests ────────────────────────────────────────
@pytest.fixture(scope="module")
def inference_data():
    """Devuelve (X_sample, preds, indices) para 100 instantes temporales espaciados."""
    returns = load_returns(str(DATA_DIR), RETURNS_FILE)
    X, _ = create_time_series_data(returns, INPUT_WINDOW, OUTPUT_WINDOW)
    # X: (n_total, INPUT_WINDOW, n_assets) — sin escalado (igual que en el notebook)

    n_total = len(X)

    # 100 índices equiespaciados; paso >= INPUT_WINDOW garantiza ausencia de solapamiento
    step = max(MIN_STEP, n_total // N_SAMPLES)
    indices = np.arange(0, n_total, step)[:N_SAMPLES]
    assert len(indices) == N_SAMPLES, (
        f"Solo se obtuvieron {len(indices)} índices con step={step}; "
        f"reduce N_SAMPLES o MIN_STEP."
    )

    X_sample = X[indices]  # (N_SAMPLES, INPUT_WINDOW, n_assets)

    model = keras.models.load_model(MODEL_PATH)
    preds = model.predict(X_sample, verbose=0)  # (N_SAMPLES, n_assets)
    return X_sample, preds, indices


# ── tests de infraestructura ───────────────────────────────────────────────

def test_model_file_exists():
    """El fichero .keras existe antes de intentar cargarlo."""
    assert MODEL_PATH.exists(), f"Modelo no encontrado: {MODEL_PATH}"


def test_output_shape(inference_data):
    """El modelo devuelve exactamente N_SAMPLES predicciones."""
    _, preds, _ = inference_data
    assert preds.shape[0] == N_SAMPLES


# ── tests de INPUTS (sanidad del setup) ───────────────────────────────────
# Si estos tests fallan, hay un bug en el propio test, no en el modelo.

def test_inputs_differ_globally(inference_data):
    """Los 100 inputs crudos no son todos iguales (std global > 0)."""
    X_sample, _, _ = inference_data
    std = X_sample.std()
    assert std > 1e-6, (
        f"std global de inputs = {std:.2e}. Los 100 inputs son idénticos — "
        "bug en la selección de índices."
    )


def test_inputs_per_asset_differ(inference_data):
    """Para cada activo, los 100 inputs varían a lo largo del tiempo."""
    X_sample, _, _ = inference_data
    # X_sample: (N_SAMPLES, INPUT_WINDOW, n_assets) → std por feature colapsando tiempo
    per_asset_std = X_sample.reshape(N_SAMPLES, -1).std(axis=0)
    n_zero = (per_asset_std <= 1e-6).sum()
    assert n_zero == 0, (
        f"{n_zero} features de input tienen std ≈ 0. "
        "Los inputs no varían — bug en la selección de índices."
    )


# ── tests de PREDICCIONES (comportamiento del modelo) ─────────────────────
# Si los tests de inputs pasan pero estos fallan, el modelo es un predictor
# constante: aprende un valor fijo por activo e ignora el input.
# Causa probable: la pérdida MAE converge a la mediana cuando la señal es débil.

def test_predictions_are_finite(inference_data):
    """Todas las predicciones son valores finitos (sin NaN ni Inf)."""
    _, preds, _ = inference_data
    assert np.all(np.isfinite(preds)), "Hay NaN o Inf en las predicciones."


def test_predictions_vary_over_time(inference_data):
    """Para cada activo, la predicción cambia a lo largo de los 100 instantes.

    Fallo => el modelo predice un valor constante por activo sin atender al input
    (predictor constante). Investiga la curva de entrenamiento o reentrena el modelo.
    """
    X_sample, preds, _ = inference_data
    per_asset_std_input = X_sample.mean(axis=1).std(axis=0)   # variación real en el input
    per_asset_std_pred  = preds.std(axis=0)                    # variación en las predicciones

    collapsed = np.where(per_asset_std_pred <= 1e-6)[0]
    n_assets  = preds.shape[1]

    diagnostics = "\n".join(
        f"  activo {i:2d}: input_std={per_asset_std_input[i]:.4f}  "
        f"pred_std={per_asset_std_pred[i]:.2e}"
        for i in collapsed
    )

    assert len(collapsed) == 0, (
        f"{len(collapsed)}/{n_assets} activos tienen predicción constante en el tiempo "
        f"(pred_std ≈ 0) pese a inputs variables.\n"
        f"Activos colapsados (input_std vs pred_std):\n{diagnostics}\n"
        "=> El modelo ha convergido a un predictor constante. "
        "Revisa las curvas de entrenamiento o reentrena el modelo."
    )


def test_first_vs_last_differ(inference_data):
    """Las predicciones del instante más antiguo y el más reciente difieren."""
    _, preds, indices = inference_data
    gap = int(indices[-1] - indices[0])
    mean_diff = float(np.abs(preds[0] - preds[-1]).mean())
    assert mean_diff > 1e-6, (
        f"Predicciones en t=0 y t={gap} son idénticas (diff media={mean_diff:.2e}). "
        "El modelo no responde al input."
    )


def test_consecutive_pairs_mostly_differ(inference_data):
    """Al menos el 90 % de los pares consecutivos tienen predicciones distintas."""
    _, preds, _ = inference_data
    diffs = np.abs(preds[1:] - preds[:-1]).mean(axis=1)
    n_different = int((diffs > 1e-8).sum())
    threshold = int(0.9 * (N_SAMPLES - 1))
    assert n_different >= threshold, (
        f"Solo {n_different}/{N_SAMPLES - 1} pares consecutivos difieren "
        f"(mínimo: {threshold}). El modelo parece ignorar el input."
    )


# ── ejecución directa ─────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
