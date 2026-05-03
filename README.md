# Taller B3-T4/T5/T6 - Redes Neuronales para Forecasting

Proyecto del bloque 3 orientado a la prediccion a futuro de activos financieros mediante redes neuronales. El objetivo es comparar arquitecturas densas, recurrentes, convolucionales y mixtas sobre distintas ventanas temporales de entrada y salida.

## Objetivo

Entrenar modelos que, a partir de una ventana historica de datos diarios de 23 activos del S&P 500, predigan el promedio futuro de cada activo durante una ventana de salida.

- Entradas: `X` con forma `(N, V, Ch)`, donde `V` es la ventana temporal de entrada y `Ch = 23` activos.
- Salidas: `y` con forma `(N, Ch)`, con el promedio futuro por activo.
- Metrica principal: `MAE`.
- Ventanas de entrada: `5`, `10`, `30` y `90` dias.
- Ventanas de salida: `1`, `5`, `30` y `90` dias.

## Datos

Los datos se guardan localmente en `data/`:

- `data/precios_close.parquet`: precios de cierre ajustados.
- `data/returns.parquet`: retornos logaritmicos diarios.
- `data/sequences.npz`: secuencias preprocesadas, si se generan.
- `data/lr_benchmark.csv`: benchmark de regresion lineal.

Los activos usados son:

`AEP`, `BA`, `CAT`, `CNP`, `CVX`, `DIS`, `DTE`, `ED`, `GD`, `GE`, `HON`, `HPQ`, `IBM`, `IP`, `JNJ`, `KO`, `KR`, `MMM`, `MO`, `MRK`, `MSI`, `PG`, `XOM`.

La carga y generacion de ventanas se centraliza en `util.py`.

## Estructura del Proyecto

```text
.
├── data/                         # Datos procesados
├── model/                        # Notebooks de modelos
│   ├── model_base_file.ipynb     # Plantilla minima para nuevos modelos
│   ├── regresion_lineal.ipynb    # Benchmark clasico de regresion lineal
│   ├── mlp_multilayer_perceptron.ipynb
│   ├── rnn-lstm.ipynb            # Red neuronal recurrente con LSTM
│   └── rnn-gru.ipynb             # Red neuronal recurrente con GRU
├── Lectura_datos.ipynb           # Lectura, limpieza y guardado de datos
├── util.py                       # Funciones comunes de datos, benchmarks y graficas
├── requirements.txt              # Dependencias Python del proyecto
├── Taller_B3_T4.pdf              # Enunciado de la practica
└── README.md
```

## Instalacion

Para ejecutar los notebooks en un entorno nuevo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Despues, abrir Jupyter desde la raiz del repositorio:

```bash
jupyter notebook
```

Los notebooks de modelos estan en `model/`. Estan preparados para importar `util.py` desde la raiz y seguir usando la carpeta raiz `data/`.

## Seguimiento de Experimentos con MLflow

Los notebooks de redes neuronales registran cada run en MLflow: parámetros, métricas por época (`train_loss`, `val_loss`), MAE final de train/validación/test, curva de perdida y el modelo serializado.

Los experimentos se almacenan en `model/mlruns/` (excluido del repositorio vía `.gitignore`).

Para visualizar los resultados, lanzar la interfaz local desde el subdirectorio `model/`:

```bash
cd model
mlflow ui
```

Esto arranca un servidor en [http://localhost:5000](http://localhost:5000) donde se pueden comparar runs, ver curvas de entrenamiento y consultar los artefactos de cada experimento.

## Funcionalidades Reutilizables

El repositorio esta organizado para que los notebooks de modelos contengan el minimo codigo posible: definir la arquitectura, entrenarla y guardar sus metricas. La carga de datos, la creacion de ventanas temporales, la particion train/test y la comparacion contra benchmarks se reutilizan desde `util.py`.

### `util.py`

Funciones principales:

- `load_precios_close()`: carga `data/precios_close.parquet` con los precios de cierre ajustados.
- `load_returns()`: carga `data/returns.parquet` con los retornos logaritmicos diarios. Usa cache para no leer el fichero repetidamente dentro de un mismo kernel.
- `create_time_series_data(data, input_window_size, output_window_size)`: transforma una serie temporal multivariante en pares `(X, y)`.
  - `X` tiene forma `(n_muestras, input_window_size, n_activos)`.
  - `y` tiene forma `(n_muestras, n_activos)` y representa la media futura de la ventana de salida.
- `get_train_test(input_window_size, output_window_size, test_size=0.1)`: carga los retornos, genera las ventanas y devuelve `X_train`, `y_train`, `X_test`, `y_test` respetando el orden temporal (`shuffle=False`).
- `save_benchmark(results_df, name)`: guarda resultados de un grid en `data/<name>.csv`.
- `load_benchmark(name="lr_benchmark")`: carga un benchmark guardado.
- `compare_to_benchmark(results_df, benchmark="lr_benchmark")`: compara los resultados de un modelo contra la regresion lineal por cada combinacion de ventanas. Devuelve el `MAE_test` del modelo, el `MAE_test_benchmark`, la diferencia absoluta (`delta`) y la diferencia porcentual (`pct_delta`).
- `plot_benchmark_comparison(results_df, benchmark="lr_benchmark", model_name="modelo")`: genera heatmaps de MAE del modelo, MAE de la regresion lineal y diferencia entre ambos.

### `model/regresion_lineal.ipynb`

Este notebook sirve como plantilla de referencia para el resto de modelos:

1. Define el grid comun de ventanas:
   - `input_windows = [5, 10, 30, 90]`
   - `output_windows = [1, 5, 30, 90]`
2. Recorre todas las combinaciones `input_window x output_window`.
3. Usa `get_train_test()` para obtener datos listos para entrenar.
4. Aplana `X` cuando el modelo lo necesita, como en `LinearRegression`.
5. Calcula `MAE_train` y `MAE_test`.
6. Construye un `results_df` con columnas estandar:
   - `input_window`
   - `output_window`
   - `MAE_train`
   - `MAE_test`
7. Guarda el resultado con `save_benchmark(results_df, "lr_benchmark")`.

Los notebooks de redes neuronales dentro de `model/` deben seguir el mismo patron, pero cambiando solo la parte especifica del modelo: arquitectura, compilacion, entrenamiento y prediccion.

Como los notebooks estan dentro de `model/` y `util.py` esta en la raiz del repositorio, los notebooks incluyen un bloque inicial que anade la raiz al `sys.path`. Asi pueden ejecutar `from util import ...` y seguir leyendo/escribiendo en la carpeta raiz `data/`.

## Modelos a Evaluar

Para cubrir el minimo exigido de 64 modelos se evaluaran, como base, 4 familias de modelos para cada una de las 16 combinaciones de ventanas:

1. MLP denso.
2. LSTM.
3. GRU.
4. Conv1D.

Como extensiones se pueden incluir:

- MLP regularizado con `Dropout`, `BatchNormalization` o regularizacion L1/L2.
- Modelos recurrentes apilados.
- Modelos bidireccionales.
- Modelos mixtos `Conv1D + GRU` o `Conv1D + LSTM`.
- Modelos con mecanismos de atencion.

## Benchmarks

El benchmark inicial es una regresion lineal multisalida entrenada para todas las combinaciones de ventanas. Sus resultados se guardan en:

```text
data/lr_benchmark.csv
```

Este benchmark sirve para comparar los modelos neuronales celda a celda en la matriz:

```text
ventana_entrada x ventana_salida
```

La comparacion contra regresion lineal es el criterio principal para interpretar si un modelo neuronal aporta valor:

- `delta < 0`: el modelo mejora a la regresion lineal en esa combinacion de ventanas.
- `delta > 0`: el modelo empeora frente a la regresion lineal.
- `pct_delta` indica el porcentaje de mejora o empeoramiento relativo.

Por tanto, no basta con reportar el MAE aislado de cada modelo. Cada notebook debe comparar sus resultados contra `lr_benchmark` para decidir si el modelo es bueno o malo respecto a una referencia simple y reproducible.

## Flujo de Trabajo

1. Cargar o regenerar los datos con `Lectura_datos.ipynb`.
2. Generar el benchmark clasico con `model/regresion_lineal.ipynb`.
3. Entrenar modelos neuronales para todas las ventanas.
4. Guardar resultados con MLflow (automático en cada notebook de modelo):
   - MAE de entrenamiento, validación y test.
   - Curvas de pérdida por epoca.
   - Número de parámetros y épocas entrenadas.
   - Modelo serializado como artefacto.
5. Generar tablas y graficas comparativas.
6. Seleccionar el mejor modelo por combinacion de ventanas.
7. Para la parte de investigacion, aplicar tecnicas de preprocesado financiero y comparar su efecto.
8. Para `output_window = 90`, construir y comparar dos carteras durante 2025:
   - cartera base sin predicciones;
   - cartera basada en predicciones del modelo.

## Resultados Esperados

El entregable debe incluir:

- Tabla de resultados por modelo y combinacion de ventanas.
- Matriz final con el mejor MAE de test para cada combinacion.
- Graficas de entrenamiento de cada modelo.
- Graficas comparativas por ventana de entrada y salida.
- Comparacion frente a modelos simples, como Buy and Hold.
- Analisis de que arquitecturas funcionan mejor segun el horizonte de prediccion.
- Resultados de las carteras para 2025.

## Ejecucion y Reutilizacion en Notebooks

### Cargar datos para cualquier modelo

En cualquier notebook de modelo basta con importar `get_train_test`:

```python
from util import get_train_test

d = get_train_test(input_window_size=10, output_window_size=5)

print(d.X_train.shape)
print(d.y_train.shape)
print(d.X_test.shape)
print(d.y_test.shape)
```

Para modelos que esperan entrada 2D, como regresion lineal o algunos modelos de scikit-learn, se puede aplanar la ventana temporal:

```python
X_train_flat = d.X_train.reshape(d.X_train.shape[0], -1)
X_test_flat = d.X_test.reshape(d.X_test.shape[0], -1)
```

Para modelos neuronales recurrentes o convolucionales, se puede usar `d.X_train` directamente con forma `(muestras, ventana, activos)`.

### Plantilla minima para un notebook de modelo

Cada notebook nuevo deberia mantener esta estructura para que los resultados sean comparables:

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error

from util import get_train_test, compare_to_benchmark, plot_benchmark_comparison

input_windows = [5, 10, 30, 90]
output_windows = [1, 5, 30, 90]

results = []

for in_w in input_windows:
    for out_w in output_windows:
        d = get_train_test(input_window_size=in_w, output_window_size=out_w)

        # Aqui va solo el codigo especifico del modelo:
        # 1. crear modelo
        # 2. entrenar con d.X_train, d.y_train
        # 3. predecir train y test

        mae_train = mean_absolute_error(d.y_train, y_pred_train)
        mae_test = mean_absolute_error(d.y_test, y_pred_test)

        results.append({
            "input_window": in_w,
            "output_window": out_w,
            "MAE_train": mae_train,
            "MAE_test": mae_test,
        })

results_df = pd.DataFrame(results)
```

Si el modelo calcula validacion, parametros o epocas, se pueden anadir columnas extra (`MAE_val`, `params`, `epochs`), manteniendo siempre las columnas base anteriores para poder comparar contra los benchmarks.

### Comparar un modelo contra regresion lineal

Esta comparacion debe hacerse al final de cada notebook de modelo. Es la forma estandar de evaluar si la arquitectura mejora o empeora el benchmark clasico.

```python
from util import compare_to_benchmark, plot_benchmark_comparison

comparison = compare_to_benchmark(results_df, benchmark="lr_benchmark")
fig = plot_benchmark_comparison(results_df, benchmark="lr_benchmark", model_name="MLP")
```

El benchmark esperado es `data/lr_benchmark.csv`, generado desde `model/regresion_lineal.ipynb`.

Interpretacion de `comparison`:

- `MAE_test`: error del modelo evaluado.
- `MAE_test_benchmark`: error de la regresion lineal para la misma ventana de entrada y salida.
- `delta`: `MAE_test - MAE_test_benchmark`. Valores negativos significan mejora.
- `pct_delta`: mejora o empeoramiento porcentual frente a regresion lineal.

## Entregables

- Repositorio GitHub con codigo, datos procesados o instrucciones de generacion, modelos, tablas y graficas.
- PDF de presentacion con:
  - matriz final de resultados;
  - reflexion sobre modelos y ventanas;
  - explicacion del preprocesado;
  - comparacion de carteras para 2025.

## Notas Metodologicas

- La particion train/test debe respetar el orden temporal.
- El test no debe usarse para seleccionar hiperparametros.
- La validacion debe salir del bloque de entrenamiento.
- Conviene fijar semillas para hacer los experimentos reproducibles.
- Los modelos deben entrenarse con perdida `MAE`, coherente con la metrica de evaluacion.
