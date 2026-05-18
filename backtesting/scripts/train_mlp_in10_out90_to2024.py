"""Entrena el MLP 4x100 GELU + Dropout + L2 (input=10, output=90) hasta 2024.

Basado en model/mlp/09_mlp_4x100_gelu_dropout_l2.ipynb para la ventana
input=10/output=90, pero usando returns_to2024.parquet como fuente de datos
para reservar 2025 como periodo de evaluación out-of-sample del backtesting.
El artefacto incluye una capa Normalization adaptada al train, de modo que
el backtesting puede seguir pasando retornos crudos al modelo.

El entrenamiento usa hasta 500 epochs por defecto, pero el artefacto final no
es la última época: se guarda el menor val_loss entre epochs cuyas predicciones
siguen variando en una muestra de validación. Esto evita seleccionar el mínimo
MAE colapsado a predictor constante.

El modelo queda guardado en:
    data/mlp/saved_models/mlp_4x100_gelu_dropout_l2_input10_output90_model.keras

Uso desde terminal:
    python backtesting/scripts/train_mlp_in10_out90_to2024.py
    MLP_EPOCHS=300 python backtesting/scripts/train_mlp_in10_out90_to2024.py

Uso desde otro script o notebook:
    from backtesting.scripts.train_mlp_in10_out90_to2024 import run
    row = run(epochs=300)
"""

import os
import sys
from pathlib import Path

import numpy as np
import keras
from keras import regularizers
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Input, Normalization
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from util import RANDOM_SEED, get_train_test  # noqa: E402

# ── hiperparámetros ────────────────────────────────────────────────────────
MODEL_NAME = "mlp_4x100_gelu_dropout_l2"
INPUT_W = 10
OUTPUT_W = 90
HIDDEN_LAYERS = 4
UNITS = 100
DROPOUT = 0.20
L2 = 1e-5
LR = 1e-4
VALIDATION_SPLIT = 0.10
RETURNS_FILE = "returns_to2024.parquet"
DEFAULT_EPOCHS = 500
PATIENCE = 30
MIN_DELTA = 1e-7
VARIATION_N_SAMPLES = 100
MIN_PRED_STD = 1e-6
MIN_FIRST_LAST_DIFF = 1e-6
MIN_PAIR_DIFF = 1e-8
MIN_PAIR_RATIO = 0.90


def build_model(input_dim: int, output_dim: int, X_adapt: np.ndarray) -> Sequential:
    """Construye el MLP usado en el notebook con normalización interna de inputs."""
    l2_reg = regularizers.l2(L2)
    normalizer = Normalization()
    normalizer.adapt(X_adapt)

    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(normalizer)
    for _ in range(HIDDEN_LAYERS):
        model.add(Dense(UNITS, activation="gelu", kernel_regularizer=l2_reg))
        model.add(Dropout(DROPOUT))
    model.add(Dense(output_dim))
    model.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=LR))
    return model


def make_variation_probe(X: np.ndarray, n_samples: int = VARIATION_N_SAMPLES) -> np.ndarray:
    """Selecciona inputs espaciados para detectar predictores constantes."""
    if len(X) <= n_samples:
        return X
    step = max(INPUT_W, len(X) // n_samples)
    indices = np.arange(0, len(X), step)[:n_samples]
    if len(indices) < n_samples:
        indices = np.linspace(0, len(X) - 1, num=n_samples, dtype=int)
    return X[indices]


class UsableBestModelCheckpoint(Callback):
    """Guarda el menor val_loss entre epochs cuyas predicciones no colapsan."""

    def __init__(self, filepath: Path, X_probe: np.ndarray):
        super().__init__()
        self.filepath = filepath
        self.X_probe = X_probe
        self.best = np.inf
        self.best_epoch: int | None = None
        self.best_stats: dict[str, float | int] = {}

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        preds = self.model.predict(self.X_probe, verbose=0)
        per_asset_std = preds.std(axis=0)
        first_last_diff = float(np.abs(preds[0] - preds[-1]).mean())
        pair_diffs = np.abs(preds[1:] - preds[:-1]).mean(axis=1)
        pair_ratio = float((pair_diffs > MIN_PAIR_DIFF).mean()) if len(pair_diffs) else 1.0
        min_pred_std = float(per_asset_std.min())

        usable = (
            min_pred_std > MIN_PRED_STD
            and first_last_diff > MIN_FIRST_LAST_DIFF
            and pair_ratio >= MIN_PAIR_RATIO
        )
        if usable and float(val_loss) < self.best:
            self.best = float(val_loss)
            self.best_epoch = epoch + 1
            self.best_stats = {
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best,
                "best_min_pred_std": min_pred_std,
                "best_first_last_diff": first_last_diff,
                "best_pair_ratio": pair_ratio,
            }
            self.model.save(self.filepath)

    def on_train_end(self, logs: dict | None = None) -> None:
        if self.best_epoch is None:
            raise RuntimeError(
                "No se encontró ningún epoch usable: todas las predicciones "
                "del probe de validación colapsaron."
            )


def run(
    epochs: int = int(os.getenv("MLP_EPOCHS", str(DEFAULT_EPOCHS))),
    batch_size: int = int(os.getenv("MLP_BATCH_SIZE", "256")),
    bar_type: str | None = None,
    preprocessing_dir: Path | None = None,
    ffd: bool = False,
) -> dict:
    """Entrena el MLP y guarda el artefacto .keras esperado por backtesting."""
    if bar_type is not None or preprocessing_dir is not None or ffd:
        raise ValueError("El trainer MLP solo soporta retornos diarios raw sin bars ni FFD.")

    print(f"Cargando datos desde {RETURNS_FILE} ...")
    d = get_train_test(
        input_window_size=INPUT_W,
        output_window_size=OUTPUT_W,
        returns_file=RETURNS_FILE,
    )

    X_train_full = d.X_train.reshape(d.X_train.shape[0], -1)
    X_test = d.X_test.reshape(d.X_test.shape[0], -1)
    y_train_full = d.y_train
    y_test = d.y_test
    val_size = int(VALIDATION_SPLIT * X_train_full.shape[0])
    if val_size <= 0 or val_size >= X_train_full.shape[0]:
        raise ValueError(
            f"VALIDATION_SPLIT={VALIDATION_SPLIT} produce val_size={val_size} "
            f"para {X_train_full.shape[0]} muestras."
        )

    X_fit = X_train_full[:-val_size]
    y_fit = y_train_full[:-val_size]
    X_val = X_train_full[-val_size:]
    y_val = y_train_full[-val_size:]
    X_val_probe = make_variation_probe(X_val)

    print(f"X_fit: {X_fit.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    np.random.seed(RANDOM_SEED)
    keras.utils.set_random_seed(RANDOM_SEED)

    model = build_model(
        input_dim=X_train_full.shape[1],
        output_dim=y_train_full.shape[1],
        X_adapt=X_fit,
    )
    model.summary()

    model_dir = PROJECT_ROOT / "data" / "mlp" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_NAME}_input{INPUT_W}_output{OUTPUT_W}_model.keras"

    usable_checkpoint = UsableBestModelCheckpoint(model_path, X_val_probe)
    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            usable_checkpoint,
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=PATIENCE,
                min_delta=MIN_DELTA,
                restore_best_weights=False,
            ),
        ],
        verbose=int(os.getenv("MLP_VERBOSE", "2")),
        shuffle=False,
    )

    model = load_model(model_path)

    y_pred_train = model.predict(X_fit, verbose=0)
    y_pred_val = model.predict(X_val, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)

    mae_train = mean_absolute_error(y_fit, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    return {
        "mae_train": mae_train,
        "mae_val": mae_val,
        "mae_test": mae_test,
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": usable_checkpoint.best_epoch,
        **usable_checkpoint.best_stats,
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
    print(f"  best epoch: {row['best_epoch']}")
    print(f"  parámetros: {row['n_params']}")

    model_path = PROJECT_ROOT / row["model_path"]
    print(f"\nModelo guardado en: {model_path}")
    print("Para cargar en backtesting:")
    print(f"  model = keras.models.load_model(r'{model_path}')")
