"""Utilidades para cargar los datos y generar ventanas train/test on-demand."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).parent / "data"
PROJECT_ROOT = Path(__file__).parent
MLFLOW_DB_PATH = PROJECT_ROOT / "model" / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "model" / "mlruns"
RANDOM_SEED = 42  # mismo seed que en Lectura_datos.ipynb

GRID_KEYS = ["input_window", "output_window"]
BENCHMARK_COLS = [*GRID_KEYS, "MAE_train", "MAE_test"]


class TrainTestData(NamedTuple):
    """Contenedor con train/test. Permite unpacking o acceso por nombre."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@lru_cache(maxsize=1)
def load_precios_close(data_dir: str = str(DATA_DIR)) -> pd.DataFrame:
    """Carga (y cachea) el DataFrame de precios de cierre ajustados."""
    return pd.read_parquet(Path(data_dir) / "precios_close.parquet")


@lru_cache(maxsize=1)
def load_returns(data_dir: str = str(DATA_DIR)) -> pd.DataFrame:
    """Carga (y cachea) el DataFrame de retornos logarítmicos."""
    return pd.read_parquet(Path(data_dir) / "returns.parquet")


def create_time_series_data(
    data: pd.DataFrame | np.ndarray,
    input_window_size: int,
    output_window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera secuencias (X, y) para series temporales multivariantes.

    X: (n_samples, input_window_size, n_features)
    y: (n_samples, n_features) — media de la ventana de salida, o el último
       valor de la ventana de entrada si output_window_size == 0.

    Implementación vectorizada con sliding_window_view: O(N) sin bucles Python.
    """
    arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
    n_total = len(arr)
    n_samples = n_total - input_window_size - output_window_size + 1
    if n_samples <= 0:
        raise ValueError(
            f"Ventanas demasiado grandes para los datos: n_total={n_total}, "
            f"input={input_window_size}, output={output_window_size}"
        )

    # X: ventanas de tamaño input_window_size sobre el eje temporal
    # sliding_window_view devuelve (n_total - w + 1, n_features, w); transponemos a (n, w, n_features)
    X_full = sliding_window_view(arr, window_shape=input_window_size, axis=0)
    X = X_full[:n_samples].transpose(0, 2, 1).copy()

    if output_window_size > 0:
        y_full = sliding_window_view(arr, window_shape=output_window_size, axis=0)
        # y[i] = mean(arr[i+input : i+input+output])  ==  y_full[i+input].mean(axis=-1)
        y = y_full[input_window_size : input_window_size + n_samples].mean(axis=-1)
    else:
        y = arr[input_window_size - 1 : input_window_size - 1 + n_samples].copy()

    return X, y


def save_benchmark(
    results_df: pd.DataFrame,
    name: str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Guarda resultados de un grid (input_window × output_window) como benchmark.

    Espera un DataFrame con columnas: input_window, output_window, MAE_train, MAE_test.
    Devuelve la ruta del CSV escrito.
    """
    missing = [c for c in BENCHMARK_COLS if c not in results_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas {missing} en results_df")
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    results_df[BENCHMARK_COLS].sort_values(GRID_KEYS).to_csv(out_path, index=False)
    return out_path


def load_benchmark(
    name: str = "lr_benchmark",
    data_dir: str | Path = DATA_DIR,
) -> pd.DataFrame:
    """Carga un benchmark por nombre desde data/<name>.csv."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe benchmark '{name}' en {path}. "
            f"Genera uno primero con save_benchmark(results_df, '{name}')."
        )
    return pd.read_csv(path)


def compare_to_benchmark(
    results_df: pd.DataFrame,
    benchmark: str = "lr_benchmark",
) -> pd.DataFrame:
    """Alinea los resultados por (input_window, output_window) frente al benchmark.

    Devuelve: input_window, output_window, MAE_test, MAE_test_benchmark,
    delta (= MAE_test − benchmark) y pct_delta (%).
    delta < 0 y pct_delta < 0 ⇒ el modelo mejora al benchmark.

    Nota: se usa "pct_delta" (y no "pct_change") para no sombrear
    el método `DataFrame.pct_change` cuando se accede como atributo.
    """
    bench = load_benchmark(benchmark)[GRID_KEYS + ["MAE_test"]].rename(
        columns={"MAE_test": "MAE_test_benchmark"}
    )
    merged = results_df[GRID_KEYS + ["MAE_test"]].merge(bench, on=GRID_KEYS, how="inner")
    merged["delta"] = merged["MAE_test"] - merged["MAE_test_benchmark"]
    merged["pct_delta"] = 100 * merged["delta"] / merged["MAE_test_benchmark"]
    return merged.sort_values(GRID_KEYS).reset_index(drop=True)


def plot_benchmark_comparison(
    results_df: pd.DataFrame,
    benchmark: str = "lr_benchmark",
    model_name: str = "modelo",
):
    """Tres heatmaps: modelo, benchmark, delta (divergente centrado en 0).

    Azul = el modelo mejora al benchmark; rojo = empeora.
    """
    comp = compare_to_benchmark(results_df, benchmark=benchmark)
    model_mat = comp.pivot(index="input_window", columns="output_window", values="MAE_test")
    bench_mat = comp.pivot(index="input_window", columns="output_window", values="MAE_test_benchmark")
    delta_mat = comp.pivot(index="input_window", columns="output_window", values="delta")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin = min(model_mat.values.min(), bench_mat.values.min())
    vmax = max(model_mat.values.max(), bench_mat.values.max())
    dlim = np.abs(delta_mat.values).max()

    panels = [
        (axes[0], model_mat, f"MAE test — {model_name}", "viridis", vmin, vmax),
        (axes[1], bench_mat, f"MAE test — {benchmark}", "viridis", vmin, vmax),
        (axes[2], delta_mat, f"Δ ({model_name} − {benchmark})", "RdBu_r", -dlim, dlim),
    ]
    for ax, mat, title, cmap, lo, hi in panels:
        im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=lo, vmax=hi)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("Ventana de salida (días)")
        ax.set_ylabel("Ventana de entrada (días)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat.values[i, j]:.4f}",
                        ha="center", va="center", color="black", fontsize=9)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def get_train_test(
    input_window_size: int,
    output_window_size: int,
    test_size: float = 0.1,
    random_state: int = RANDOM_SEED,
    data_dir: str | Path = DATA_DIR,
) -> TrainTestData:
    """Devuelve train/test listos para un par (input_window, output_window).

    Los retornos se cachean entre llamadas, así que es eficiente barrer varias
    combinaciones de ventanas en bucle.
    """
    returns = load_returns(str(data_dir))
    X, y = create_time_series_data(returns, input_window_size, output_window_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    return TrainTestData(X_train, y_train, X_test, y_test)

def plot_training_curve(history, show=False):
    """Grafica la pérdida de entrenamiento y validación a lo largo de las épocas.

    Recibe el objeto `History` devuelto por `model.fit()` y genera una figura
    con las curvas `loss` y `val_loss`. Si `show=True`, llama a `plt.show()`;
    en caso contrario, devuelve la figura sin mostrarla (útil en notebooks o
    cuando se quiere guardar el plot más adelante).
    """
    fig = plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Evolución de la pérdida')
    plt.legend()
    plt.grid(True)
    if show:
        plt.tight_layout()
        plt.show()
    return fig


def configure_mlflow(
    experiment_name: str,
    tracking_db_path: str | Path = MLFLOW_DB_PATH,
    artifact_root: str | Path = MLFLOW_ARTIFACT_ROOT,
):
    """Configura MLflow para usar la base SQLite compartida del proyecto.

    Si el experimento no existe, lo crea con artefactos locales bajo
    model/mlruns/. Esto evita depender de rutas absolutas de otros equipos.
    Devuelve el modulo `mlflow` ya configurado para que los notebooks puedan
    usarlo sin repetir boilerplate.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_db_path = Path(tracking_db_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    tracking_uri = f"sqlite:///{tracking_db_path.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        safe_name = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in experiment_name
        )
        artifact_location = (artifact_root / safe_name).as_uri()
        client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(experiment_name)
    return mlflow


def _local_artifact_path(artifact_uri: str) -> Path | None:
    """Devuelve una ruta local para URIs de artefactos locales."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        return None
    return Path(artifact_uri)


def _can_write_artifacts(artifact_uri: str) -> bool:
    """Comprueba si el artifact_uri actual permite escribir artefactos."""
    artifact_path = _local_artifact_path(artifact_uri)
    if artifact_path is None:
        return False

    try:
        artifact_path.mkdir(parents=True, exist_ok=True)
        probe = artifact_path / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False

    return True


def log_keras_grid_run(
    mlflow,
    model,
    history,
    run_name: str,
    model_name: str,
    input_window: int,
    output_window: int,
    metrics_row: dict,
    train_shape: tuple[int, ...],
    test_shape: tuple[int, ...],
    output_dim: int,
    batch_size: int,
    learning_rate: float,
    validation_split: float,
    extra_params: dict | None = None,
    log_model_artifact: bool = True,
) -> None:
    """Registra en MLflow una ejecucion de Keras para una celda del grid.

    Mantiene nombres de metricas compatibles con `regresion_lineal.ipynb`:
    train_mae, val_mae y test_mae. Tambien registra las curvas por epoca para
    poder compararlas desde la UI de MLflow.
    """
    existing_runs = mlflow.search_runs(
        filter_string=f'tags.mlflow.runName = "{run_name}"'
    )
    if not existing_runs.empty:
        for run_id in existing_runs["run_id"]:
            mlflow.delete_run(run_id)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", "MLP")
        mlflow.set_tag("model_name", model_name)

        params = {
            "model_type": "Dense_Neural_Network",
            "model_name": model_name,
            "input_window_size": input_window,
            "output_window_size": output_window,
            "input_dim": train_shape[1],
            "output_dim": output_dim,
            "num_train_samples": train_shape[0],
            "num_test_samples": test_shape[0],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "epochs": metrics_row["epochs"],
            "n_params": metrics_row["n_params"],
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        for epoch, (loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"]),
            start=1,
        ):
            mlflow.log_metric("train_loss", float(loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)

        mlflow.log_metric("train_mae", float(metrics_row["MAE_train"]))
        mlflow.log_metric("val_mae", float(metrics_row["MAE_val"]))
        mlflow.log_metric("test_mae", float(metrics_row["MAE_test"]))

        artifact_uri = mlflow.get_artifact_uri()
        if _can_write_artifacts(artifact_uri):
            fig = plot_training_curve(history)
            mlflow.log_figure(fig, "plots/loss_curve.png")
            plt.close(fig)

            if log_model_artifact:
                mlflow.keras.log_model(model, name=f"{model_name}_model")
        else:
            mlflow.set_tag("artifacts_logged", "false")
            mlflow.set_tag("artifact_uri", artifact_uri)
