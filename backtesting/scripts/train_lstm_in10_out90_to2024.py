"""Entrena el LSTM (input=10, output=90) sobre datos hasta 2024.

Equivalente a model/rnn/lstm/rnn-lstm-input10-output90.ipynb pero usando
returns_to2024.parquet como fuente de datos, reservando 2025 como
periodo de evaluación out-of-sample para el backtesting.

Hiperparámetros fijados según el mejor modelo del notebook:
    lstm_layers  = 2
    units        = 128
    dropout      = 0.0
    learning_rate= 1e-4
    batch_size   = 128

El modelo queda guardado en:
    data/rnn/saved_models/rnn_lstm_input10_output90_model.keras

AVISO: sobreescribe el modelo guardado por el notebook original si existe.
Si necesitas conservar ambos, renombra el fichero .keras antes de ejecutar.

Uso desde terminal:
    python backtesting/scripts/train_lstm_in10_out90_to2024.py
    LSTM_EPOCHS=300 python backtesting/scripts/train_lstm_in10_out90_to2024.py

Uso desde otro script o notebook:
    from backtesting.scripts.train_lstm_in10_out90_to2024 import run
    row = run(epochs=300)
"""

import os
import sys
from pathlib import Path

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from util import get_train_test, RANDOM_SEED  # noqa: E402

# ── hiperparámetros ────────────────────────────────────────────────────────
INPUT_W      = 10
OUTPUT_W     = 90
LSTM_LAYERS  = 2
UNITS        = 128
DROPOUT      = 0.0
LR           = 1e-4
PATIENCE     = 10
RETURNS_FILE = "returns_to2024.parquet"


def run(
    epochs: int = int(os.getenv("LSTM_EPOCHS", "200")),
    batch_size: int = int(os.getenv("LSTM_BATCH_SIZE", "128")),
    bar_type: str | None = None,
    preprocessing_dir: Path | None = None,
    ffd: bool = False,
) -> dict:
    """Entrena el LSTM (input=10, output=90) sobre returns_to2024.parquet.

    bar_type ("time", "count", "volume", "dollar") carga los NPZ pre-generados
    por 03_build_preprocessed_sequences.py en lugar del parquet original.
    preprocessing_dir permite sobreescribir la ruta base de los NPZ.
    ffd activa la diferenciación fraccionaria sobre las series antes del entrenamiento.

    Returns:
        dict con MAE train/val/test, épocas entrenadas, nº parámetros y ruta del modelo.
    """
    if bar_type is not None:
        print(f"Cargando datos desde NPZ — bar_type={bar_type} ...")
    else:
        print(f"Cargando datos desde {RETURNS_FILE} ...")
    d = get_train_test(
        input_window_size=INPUT_W,
        output_window_size=OUTPUT_W,
        returns_file=RETURNS_FILE,
        bar_type=bar_type,
        preprocessing_dir=preprocessing_dir,
        ffd=ffd,
    )

    val_size  = int(0.10 * d.X_train.shape[0])
    X_val     = d.X_train[-val_size:]
    y_val     = d.y_train[-val_size:]
    X_train   = d.X_train[:-val_size]
    y_train   = d.y_train[:-val_size]
    X_test    = d.X_test
    y_test    = d.y_test

    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    for i in range(LSTM_LAYERS):
        return_seq = (i < LSTM_LAYERS - 1)
        model.add(LSTM(UNITS, return_sequences=return_seq, dropout=DROPOUT))
    model.add(Dense(y_train.shape[1]))
    model.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=LR))
    model.summary()

    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
    )

    mae_train = mean_absolute_error(y_train, model.predict(X_train, verbose=0))  # type: ignore[arg-type]
    mae_val   = mean_absolute_error(y_val,   model.predict(X_val,   verbose=0))  # type: ignore[arg-type]
    mae_test  = mean_absolute_error(y_test,  model.predict(X_test,  verbose=0))  # type: ignore[arg-type]

    model_dir  = PROJECT_ROOT / "data" / "rnn" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bar_suffix = f"_{bar_type}" if bar_type is not None else ""
    model_path = model_dir / f"rnn_lstm_input{INPUT_W}_output{OUTPUT_W}{bar_suffix}_model.keras"
    model.save(model_path)

    return {
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mae_test": mae_test,
        "epochs_trained": len(history.history["loss"]),
        "n_params": model.count_params(),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }


if __name__ == "__main__":
    row = run()

    print("\nResultados:")
    print(f"  MAE train : {row['mae_train']:.6f}")
    print(f"  MAE val   : {row['mae_val']:.6f}")
    print(f"  MAE test  : {row['mae_test']:.6f}")
    print(f"  épocas    : {row['epochs_trained']}")
    print(f"  parámetros: {row['n_params']}")

    model_path = PROJECT_ROOT / row["model_path"]
    print(f"\nModelo guardado en: {model_path}")
    print("Para cargar en backtesting:")
    print(f"  model = keras.models.load_model(r'{model_path}')")
