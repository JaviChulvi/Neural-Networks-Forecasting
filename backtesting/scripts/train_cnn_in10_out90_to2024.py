"""Entrena el CNN Deep Conv1D (input=10, output=90) sobre datos hasta 2024.

Equivalente a model/cnn/cnn_input10_output90.ipynb pero usando
returns_to2024.parquet como fuente de datos, reservando 2025 como
periodo de evaluación out-of-sample para el backtesting.

El modelo queda guardado en:
    data/cnn/saved_models/cnn_input10_output90_model.keras

AVISO: sobreescribe el modelo guardado por el notebook original si existe.
Si necesitas conservar ambos, renombra el fichero .keras antes de ejecutar.

Uso desde terminal:
    python backtesting/scripts/train_cnn_in10_out90_to2024.py
    CNN_EPOCHS=50 FORCE_RETRAIN=1 python backtesting/scripts/train_cnn_in10_out90_to2024.py

Uso desde otro script o notebook:
    from backtesting.scripts.train_cnn_in10_out90_to2024 import run
    row = run(epochs=50, force=True)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model" / "cnn"))

from cnn_utils import train_cnn_window


def run(
    epochs: int = int(os.getenv("CNN_EPOCHS", "120")),
    batch_size: int = int(os.getenv("CNN_BATCH_SIZE", "128")),
    force: bool = os.getenv("FORCE_RETRAIN", "1") == "1",
    bar_type: str | None = None,
    preprocessing_dir: Path | None = None,
    ffd: bool = False,
) -> dict:
    """Entrena el CNN (input=10, output=90) sobre returns_to2024.parquet.

    bar_type ("time", "count", "volume", "dollar") carga los NPZ pre-generados
    por 03_build_preprocessed_sequences.py en lugar del parquet original.
    preprocessing_dir permite sobreescribir la ruta base de los NPZ.
    ffd activa la diferenciación fraccionaria sobre las series antes del entrenamiento.

    Returns:
        dict con métricas y la ruta del modelo guardado.
    """
    row = train_cnn_window(
        input_window=10,
        output_window=90,
        epochs=epochs,
        batch_size=batch_size,
        force=force,
        returns_file="returns_to2024.parquet",
        bar_type=bar_type,
        preprocessing_dir=preprocessing_dir,
        ffd=ffd,
    )
    return row


if __name__ == "__main__":
    row = run()

    print("\nResultados:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    model_path = PROJECT_ROOT / row["model_path"]
    print(f"\nModelo guardado en: {model_path}")
    print("Para cargar en backtesting:")
    print(f"  model = keras.models.load_model(r'{model_path}')")
