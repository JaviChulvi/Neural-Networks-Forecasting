"""Entrena el CNN Deep Conv1D (input=30, output=90) sobre datos hasta 2024.

Usa returns_to2024.parquet como fuente de datos, reservando 2025 como
periodo de evaluación out-of-sample para el backtesting.

El modelo queda guardado en:
    data/cnn/saved_models/cnn_input30_output90_model.keras

Uso:
    python backtesting/scripts/train_cnn_in30_out90_to2024.py
    CNN_EPOCHS=50 FORCE_RETRAIN=1 python backtesting/scripts/train_cnn_in30_out90_to2024.py
"""

import os
import sys
from pathlib import Path

import keras

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model" / "cnn"))

from cnn_utils import train_cnn_window

EPOCHS = int(os.getenv("CNN_EPOCHS", "120"))
BATCH_SIZE = int(os.getenv("CNN_BATCH_SIZE", "128"))
FORCE_RETRAIN = os.getenv("FORCE_RETRAIN", "1") == "1"

row = train_cnn_window(
    input_window=30,
    output_window=90,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    force=FORCE_RETRAIN,
    returns_file="returns_to2024.parquet",
)

print("\nResultados:")
for k, v in row.items():
    print(f"  {k}: {v}")

model_path = PROJECT_ROOT / row["model_path"]
print(f"\nModelo guardado en: {model_path}")
print("Para cargar en backtesting:")
print(f"  model = keras.models.load_model(r'{model_path}')")
