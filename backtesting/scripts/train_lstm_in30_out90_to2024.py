"""Entrena el LSTM (input=30, output=90) sobre datos hasta 2024.

Equivalente a model/rnn/lstm/rnn-lstm-input30-output90.ipynb pero usando
returns_to2024.parquet como fuente de datos, reservando 2025 como
periodo de evaluación out-of-sample para el backtesting.

Hiperparámetros fijados según el mejor modelo del notebook:
    lstm_layers  = 1
    units        = 32
    dropout      = 0.2
    learning_rate= 1e-4
    batch_size   = 256

El modelo queda guardado en:
    data/rnn/saved_models/rnn_lstm_input30_output90_model.keras

AVISO: sobreescribe el modelo guardado por el notebook original si existe.
Si necesitas conservar ambos, renombra el fichero .keras antes de ejecutar.

Uso:
    python backtesting/scripts/train_lstm_in30_out90_to2024.py
    LSTM_EPOCHS=300 python backtesting/scripts/train_lstm_in30_out90_to2024.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from util import get_train_test, RANDOM_SEED  # noqa: E402

# ── hiperparámetros ────────────────────────────────────────────────────────
INPUT_W      = 30
OUTPUT_W     = 90
LSTM_LAYERS  = 1
UNITS        = 32
DROPOUT      = 0.2
LR           = 1e-4
BATCH_SIZE   = int(os.getenv("LSTM_BATCH_SIZE", "256"))
EPOCHS       = int(os.getenv("LSTM_EPOCHS", "200"))
PATIENCE     = 10
RETURNS_FILE = "returns_to2024.parquet"

# ── datos ──────────────────────────────────────────────────────────────────
print(f"Cargando datos desde {RETURNS_FILE} ...")
d = get_train_test(
    input_window_size=INPUT_W,
    output_window_size=OUTPUT_W,
    returns_file=RETURNS_FILE,
)

val_size  = int(0.10 * d.X_train.shape[0])
X_val     = d.X_train[-val_size:]
y_val     = d.y_train[-val_size:]
X_train   = d.X_train[:-val_size]
y_train   = d.y_train[:-val_size]
X_test    = d.X_test
y_test    = d.y_test

print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

# ── modelo ─────────────────────────────────────────────────────────────────
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

# ── entrenamiento ──────────────────────────────────────────────────────────
es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
)

# ── evaluación ─────────────────────────────────────────────────────────────
mae_train = mean_absolute_error(y_train, model.predict(X_train, verbose=0))
mae_val   = mean_absolute_error(y_val,   model.predict(X_val,   verbose=0))
mae_test  = mean_absolute_error(y_test,  model.predict(X_test,  verbose=0))

print("\nResultados:")
print(f"  MAE train : {mae_train:.6f}")
print(f"  MAE val   : {mae_val:.6f}")
print(f"  MAE test  : {mae_test:.6f}")
print(f"  épocas    : {len(history.history['loss'])}")
print(f"  parámetros: {model.count_params()}")

# ── guardado ───────────────────────────────────────────────────────────────
model_dir  = PROJECT_ROOT / "data" / "rnn" / "saved_models"
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / f"rnn_lstm_input{INPUT_W}_output{OUTPUT_W}_model.keras"
model.save(model_path)

print(f"\nModelo guardado en: {model_path}")
print("Para cargar en backtesting:")
print(f"  model = keras.models.load_model(r'{model_path}')")
