import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "util.py").exists():
    PROJECT_ROOT = next(p for p in PROJECT_ROOT.parents if (p / "util.py").exists())

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import keras
from keras import backend as K
from keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    SpatialDropout1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    Dense,
    Dropout,
)
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from util import get_train_test, RANDOM_SEED

try:
    from util import configure_mlflow
except Exception:
    import mlflow

    def configure_mlflow(experiment_name: str):
        tracking_uri = f"sqlite:///{PROJECT_ROOT / 'model' / 'mlflow.db'}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        return mlflow

DATA_OUT = PROJECT_ROOT / "data" / "cnn"
RESULTS_DIR = DATA_OUT / "window_results"
HISTORY_DIR = DATA_OUT / "history"
PLOTS_DIR = DATA_OUT / "plots"
MODELS_DIR = DATA_OUT / "saved_models"

for p in [DATA_OUT, RESULTS_DIR, HISTORY_DIR, PLOTS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

INPUT_WINDOWS = [5, 10, 30, 90]
OUTPUT_WINDOWS = [1, 5, 30, 90]

np.random.seed(RANDOM_SEED)
keras.utils.set_random_seed(RANDOM_SEED)


def split_train_val(X_train, y_train, val_ratio=0.10):
    val_size = int(len(X_train) * val_ratio)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    return X_train_final, y_train_final, X_val, y_val


def scale_X_only(X_train, X_val, X_test):
    n_train, window, n_assets = X_train.shape
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(n_train, -1)
    X_val_2d = X_val.reshape(n_val, -1)
    X_test_2d = X_test.reshape(n_test, -1)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, window, n_assets)
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, window, n_assets)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, window, n_assets)

    return X_train_scaled, X_val_scaled, X_test_scaled


def build_deep_cnn(input_window, n_assets):
    inputs = Input(shape=(input_window, n_assets))

    x = Conv1D(filters=64, kernel_size=3, padding="causal", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.10)(x)

    x = Conv1D(filters=64, kernel_size=5, padding="causal", dilation_rate=2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.10)(x)

    x = Conv1D(filters=128, kernel_size=3, padding="causal", dilation_rate=4, activation="relu")(x)
    x = BatchNormalization()(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = Concatenate()([avg_pool, max_pool])

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.25)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.15)(x)

    outputs = Dense(n_assets, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss="mae",
        metrics=["mae"],
    )

    return model


def save_history_and_plot(history, input_window, output_window):
    hist = pd.DataFrame(history.history)

    history_path = HISTORY_DIR / f"cnn_input{input_window}_output{output_window}_history.csv"
    plot_path = PLOTS_DIR / f"cnn_input{input_window}_output{output_window}_loss_curve.png"

    hist.to_csv(history_path, index=False)

    plt.figure(figsize=(8, 4))
    plt.plot(hist["loss"], label="Train loss")
    plt.plot(hist["val_loss"], label="Validation loss")
    plt.title(f"CNN Deep Conv1D - input={input_window}, output={output_window}")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return history_path, plot_path


def train_cnn_window(input_window, output_window, epochs=120, batch_size=128, force=False, returns_file="returns.parquet", relative=False):
    suffix = "_rel" if relative else ""
    result_path = RESULTS_DIR / f"cnn_input{input_window}_output{output_window}{suffix}_results.csv"

    if result_path.exists() and not force:
        print(f"SKIP existing result: {result_path}")
        return pd.read_csv(result_path).iloc[0].to_dict()

    print("")
    print("=" * 80)
    print(f"Training CNN Deep Conv1D | input={input_window} | output={output_window} | relative={relative}")
    print("=" * 80)

    K.clear_session()
    keras.utils.set_random_seed(RANDOM_SEED)

    d = get_train_test(
        input_window_size=input_window,
        output_window_size=output_window,
        returns_file=returns_file,
        relative=relative,
    )

    X_train_raw, y_train_raw = d.X_train, d.y_train
    X_test_raw, y_test = d.X_test, d.y_test

    X_train_raw, y_train, X_val_raw, y_val = split_train_val(X_train_raw, y_train_raw)
    X_train, X_val, X_test = scale_X_only(X_train_raw, X_val_raw, X_test_raw)

    n_assets = X_train.shape[2]
    model = build_deep_cnn(input_window, n_assets)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=18, min_delta=1e-6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_val = model.predict(X_val, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    params = model.count_params()
    epochs_trained = len(history.history["loss"])

    history_path, plot_path = save_history_and_plot(history, input_window, output_window)

    model_path = MODELS_DIR / f"cnn_input{input_window}_output{output_window}{suffix}_model.keras"
    model.save(model_path)
    print(f"Model saved: {model_path}")

    row = {
        "model": "CNN_Deep_Conv1D",
        "input_window": input_window,
        "output_window": output_window,
        "relative_target": relative,
        "MAE_train": mae_train,
        "MAE_val": mae_val,
        "MAE_test": mae_test,
        "params": params,
        "epochs_trained": epochs_trained,
        "batch_size": batch_size,
        "learning_rate": 3e-4,
        "history_path": str(history_path.relative_to(PROJECT_ROOT)),
        "plot_path": str(plot_path.relative_to(PROJECT_ROOT)),
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
    }

    pd.DataFrame([row]).to_csv(result_path, index=False)

    mlflow = configure_mlflow("cnn_deep_conv1d_grid")
    run_name = f"cnn_deep_input{input_window}_output{output_window}{suffix}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": "CNN_Deep_Conv1D",
            "input_window": input_window,
            "output_window": output_window,
            "relative_target": relative,
            "batch_size": batch_size,
            "learning_rate": 3e-4,
            "params": params,
        })
        mlflow.log_metrics({
            "MAE_train": mae_train,
            "MAE_val": mae_val,
            "MAE_test": mae_test,
            "epochs_trained": epochs_trained,
        })
        hist_df = pd.DataFrame(history.history)
        for epoch_idx, hist_row in hist_df.iterrows():
            mlflow.log_metric("loss", float(hist_row["loss"]), step=int(epoch_idx))
            mlflow.log_metric("val_loss", float(hist_row["val_loss"]), step=int(epoch_idx))
        mlflow.log_artifact(str(history_path), artifact_path="history")
        mlflow.log_artifact(str(plot_path), artifact_path="plots")

    print("Saved:", result_path)
    return row


def aggregate_cnn_grid_results():
    result_files = sorted(RESULTS_DIR.glob("cnn_input*_output*_results.csv"))
    if not result_files:
        raise FileNotFoundError("No CNN window result files found.")

    results = pd.concat([pd.read_csv(p) for p in result_files], ignore_index=True)
    results = results.sort_values(["input_window", "output_window"]).reset_index(drop=True)

    all_results_path = DATA_OUT / "cnn_all_results.csv"
    results.to_csv(all_results_path, index=False)

    lr = pd.read_csv(PROJECT_ROOT / "data" / "lr_benchmark.csv")
    comparison = results.merge(
        lr.rename(columns={"MAE_train": "LR_MAE_train", "MAE_test": "LR_MAE_test"}),
        on=["input_window", "output_window"],
        how="left",
    )
    comparison["delta_vs_lr"] = comparison["MAE_test"] - comparison["LR_MAE_test"]
    comparison["pct_delta_vs_lr"] = comparison["delta_vs_lr"] / comparison["LR_MAE_test"] * 100

    comparison_path = DATA_OUT / "cnn_comparison_vs_lr.csv"
    comparison.to_csv(comparison_path, index=False)

    matrix = results.pivot(index="input_window", columns="output_window", values="MAE_test")
    matrix_path = DATA_OUT / "cnn_test_mae_matrix.csv"
    matrix.to_csv(matrix_path)

    report_path = PROJECT_ROOT / "model" / "cnn" / "cnn_vs_lr_report.md"
    report_lines = [
        "# CNN Deep Conv1D vs Linear Regression",
        "",
        "This report summarizes the CNN Deep Conv1D results across the input/output window grid.",
        "",
        "Negative values in pct_delta_vs_lr indicate that the CNN improves over the linear regression benchmark.",
        "",
        "## Results",
        "",
        comparison[[
            "input_window", "output_window", "MAE_train", "MAE_val", "MAE_test",
            "LR_MAE_test", "pct_delta_vs_lr", "params", "epochs_trained"
        ]].to_markdown(index=False),
        "",
        "## MAE test matrix",
        "",
        matrix.to_markdown(),
        "",
    ]

    search_result = PROJECT_ROOT / "data" / "cnn_search" / "best_cnn_optimized_result.csv"
    if search_result.exists():
        report_lines += [
            "## Additional hyperparameter search",
            "",
            "A random-search hyperparameter experiment was also performed for the 30 input / 5 output window.",
            "",
            pd.read_csv(search_result).to_markdown(index=False),
            "",
        ]

    report_path.write_text("\n".join(report_lines))
    print("Saved:", all_results_path)
    print("Saved:", comparison_path)
    print("Saved:", matrix_path)
    print("Saved:", report_path)
    return results, comparison, matrix
