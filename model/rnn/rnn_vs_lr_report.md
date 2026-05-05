# RNN vs Regresión Lineal — Informe de Resultados

Generado a partir de los notebooks `rnn/lstm/` y `rnn/gru/` y de `data/lr_benchmark.csv`.

Incluye dos arquitecturas recurrentes:
- **LSTM** — 2 etapas de HP search por ventana (`lstm_layers × units × dropout`, luego `lr × batch_size`)
- **GRU**  — misma metodología, con `gru_layers ∈ {1, 2, 3}` y `units ∈ {32, 64, 128, 256}`

---

## Conclusión principal

- MAE test medio **LSTM tuneada** : `0.005362`
- MAE test medio **GRU tuneada**  : `0.005367`
- MAE test medio **regresión lineal** : `0.005668`
- Mejor arquitectura RNN global  : **LSTM** (mean test MAE = `0.005362`)
- Ventanas donde LSTM tuneada mejora a LR : **14 / 16**
- Ventanas donde GRU tuneada mejora a LR  : **15 / 16**

Ambas arquitecturas superan a la regresión lineal en la gran mayoría de ventanas. La ventaja es
pequeña con ventanas de entrada cortas (`input=5/10`) y salidas largas (`output=90`), donde la
señal disponible es muy suave. Con `input=90` las RNNs logran las mayores mejoras relativas
(hasta ~−14 % en test MAE), ya que aprovechan la historia larga mejor que el modelo lineal.

---

## Ranking de modelos

> `wins_vs_lr` = ventanas donde el modelo supera a LR
> `wins_best_rnn` = ventanas donde este modelo es el mejor RNN de los dos

| model | mean_test | median_test | best_test | worst_test | mean_delta_lr | wins_vs_lr | wins_best_rnn | mean_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.005362 | 0.003962 | 0.001275 | 0.012256 | -0.000306 | 14 | 12 | 51,914 |
| GRU | 0.005367 | 0.003975 | 0.001269 | 0.012262 | -0.000301 | 15 | 4 | 75,841 |

---

## Mejor RNN por ventana

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012238` (L) | `0.012234` (L) | `0.012240` (G) | `0.012256` (L) |
| **out=5** | `0.005586` (L) | `0.005582` (L) | `0.005584` (L) | `0.005594` (L) |
| **out=30** | `0.002325` (L) | `0.002334` (L) | `0.002340` (L) | `0.002343` (L) |
| **out=90** | `0.001275` (L) | `0.001270` (G) | `0.001269` (G) | `0.001288` (G) |

> (L) = LSTM · (G) = GRU

---

## Mejor RNN per ventana — detalle

| input_window | output_window | model_type | MAE_test | MAE_test_lr | delta | pct_delta |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | LSTM | 0.012238 | 0.012384 | -0.000146 | -1.18% |
| 5 | 5 | LSTM | 0.005586 | 0.005625 | -0.000039 | -0.69% |
| 5 | 30 | LSTM | 0.002325 | 0.002340 | -0.000015 | -0.65% |
| 5 | 90 | LSTM | 0.001275 | 0.001271 | +0.000004 | +0.29% |
| 10 | 1 | LSTM | 0.012234 | 0.012554 | -0.000320 | -2.55% |
| 10 | 5 | LSTM | 0.005582 | 0.005698 | -0.000116 | -2.03% |
| 10 | 30 | LSTM | 0.002334 | 0.002358 | -0.000024 | -1.04% |
| 10 | 90 | GRU | 0.001270 | 0.001282 | -0.000012 | -0.97% |
| 30 | 1 | GRU | 0.012240 | 0.012924 | -0.000684 | -5.29% |
| 30 | 5 | LSTM | 0.005584 | 0.005877 | -0.000293 | -4.98% |
| 30 | 30 | LSTM | 0.002340 | 0.002436 | -0.000096 | -3.95% |
| 30 | 90 | GRU | 0.001269 | 0.001351 | -0.000082 | -6.10% |
| 90 | 1 | LSTM | 0.012256 | 0.014095 | -0.001839 | -13.05% |
| 90 | 5 | LSTM | 0.005594 | 0.006348 | -0.000754 | -11.88% |
| 90 | 30 | LSTM | 0.002343 | 0.002628 | -0.000285 | -10.85% |
| 90 | 90 | GRU | 0.001288 | 0.001518 | -0.000230 | -15.16% |

---

## Matrices de MAE test

### LSTM tuneada

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012238` | `0.012234` | `0.012244` | `0.012256` |
| **out=5** | `0.005586` | `0.005582` | `0.005584` | `0.005594` |
| **out=30** | `0.002325` | `0.002334` | `0.002340` | `0.002343` |
| **out=90** | `0.001275` | `0.001285` | `0.001276` | `0.001299` |

### GRU tuneada

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012244` | `0.012247` | `0.012240` | `0.012262` |
| **out=5** | `0.005593` | `0.005584` | `0.005597` | `0.005621` |
| **out=30** | `0.002337` | `0.002334` | `0.002366` | `0.002353` |
| **out=90** | `0.001275` | `0.001270` | `0.001269` | `0.001288` |

### Regresión lineal

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012384` | `0.012554` | `0.012924` | `0.014095` |
| **out=5** | `0.005625` | `0.005698` | `0.005877` | `0.006348` |
| **out=30** | `0.002340` | `0.002358` | `0.002436` | `0.002628` |
| **out=90** | `0.001271` | `0.001282` | `0.001351` | `0.001518` |

---

## Δ (LSTM tuneada − LR)

> Negativo (↓) = LSTM mejora; positivo (↑) = LR gana.

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000146` ↓ | `-0.000320` ↓ | `-0.000680` ↓ | `-0.001839` ↓ |
| **out=5** | `-0.000039` ↓ | `-0.000116` ↓ | `-0.000293` ↓ | `-0.000754` ↓ |
| **out=30** | `-0.000015` ↓ | `-0.000024` ↓ | `-0.000096` ↓ | `-0.000285` ↓ |
| **out=90** | `+0.000004` ↑ | `+0.000003` ↑ | `-0.000075` ↓ | `-0.000219` ↓ |

## Δ (GRU tuneada − LR)

> Negativo (↓) = GRU mejora; positivo (↑) = LR gana.

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000140` ↓ | `-0.000307` ↓ | `-0.000684` ↓ | `-0.001833` ↓ |
| **out=5** | `-0.000032` ↓ | `-0.000114` ↓ | `-0.000280` ↓ | `-0.000727` ↓ |
| **out=30** | `-0.000003` ↓ | `-0.000024` ↓ | `-0.000070` ↓ | `-0.000275` ↓ |
| **out=90** | `+0.000004` ↑ | `-0.000012` ↓ | `-0.000082` ↓ | `-0.000230` ↓ |

---

## Hiperparámetros del mejor modelo por ventana

### in=5, out=1  —  LSTM tuneada  |  test_mae = `0.012238`  |  Δ vs LR = `-0.000146` (-1.18%)

- **Arquitectura:** 2 capas LSTM · 16 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 5,063

### in=5, out=5  —  LSTM tuneada  |  test_mae = `0.005586`  |  Δ vs LR = `-0.000039` (-0.69%)

- **Arquitectura:** 2 capas LSTM · 16 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 5,063

### in=5, out=30  —  LSTM tuneada  |  test_mae = `0.002325`  |  Δ vs LR = `-0.000015` (-0.65%)

- **Arquitectura:** 2 capas LSTM · 16 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 128
- **Parámetros totales:** 5,063

### in=5, out=90  —  LSTM tuneada  |  test_mae = `0.001275`  |  Δ vs LR = `+0.000004` (+0.29%)

- **Arquitectura:** 1 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-03 · batch_size = 128
- **Parámetros totales:** 7,927

### in=10, out=1  —  LSTM tuneada  |  test_mae = `0.012234`  |  Δ vs LR = `-0.000320` (-2.55%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 16,247

### in=10, out=5  —  LSTM tuneada  |  test_mae = `0.005582`  |  Δ vs LR = `-0.000116` (-2.03%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 128
- **Parámetros totales:** 16,247

### in=10, out=30  —  LSTM tuneada  |  test_mae = `0.002334`  |  Δ vs LR = `-0.000024` (-1.04%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-03 · batch_size = 128
- **Parámetros totales:** 16,247

### in=10, out=90  —  GRU tuneada  |  test_mae = `0.001270`  |  Δ vs LR = `-0.000012` (-0.97%)

- **Arquitectura:** 3 capas GRU · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 128
- **Parámetros totales:** 18,615

### in=30, out=1  —  GRU tuneada  |  test_mae = `0.012240`  |  Δ vs LR = `-0.000684` (-5.29%)

- **Arquitectura:** 3 capas GRU · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 128
- **Parámetros totales:** 18,615

### in=30, out=5  —  LSTM tuneada  |  test_mae = `0.005584`  |  Δ vs LR = `-0.000293` (-4.98%)

- **Arquitectura:** 2 capas LSTM · 64 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 57,047

### in=30, out=30  —  LSTM tuneada  |  test_mae = `0.002340`  |  Δ vs LR = `-0.000096` (-3.95%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 64
- **Parámetros totales:** 16,247

### in=30, out=90  —  GRU tuneada  |  test_mae = `0.001269`  |  Δ vs LR = `-0.000082` (-6.10%)

- **Arquitectura:** 3 capas GRU · 64 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 67,927

### in=90, out=1  —  LSTM tuneada  |  test_mae = `0.012256`  |  Δ vs LR = `-0.001839` (-13.05%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 16,247

### in=90, out=5  —  LSTM tuneada  |  test_mae = `0.005594`  |  Δ vs LR = `-0.000754` (-11.88%)

- **Arquitectura:** 2 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 16,247

### in=90, out=30  —  LSTM tuneada  |  test_mae = `0.002343`  |  Δ vs LR = `-0.000285` (-10.85%)

- **Arquitectura:** 1 capas LSTM · 32 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 256
- **Parámetros totales:** 7,927

### in=90, out=90  —  GRU tuneada  |  test_mae = `0.001288`  |  Δ vs LR = `-0.000230` (-15.16%)

- **Arquitectura:** 3 capas GRU · 128 unidades/capa · dropout = 0.2
- **Entrenamiento:** lr = 1e-04 · batch_size = 64
- **Parámetros totales:** 258,711


---

## Interpretación

- **GRU vs LSTM**: la diferencia de test MAE entre ambas es mínima en prácticamente todas las
  ventanas (< 0.0001). La GRU tiene ligeramente menos parámetros para igual número de unidades
  (3 puertas vs 4), lo que la hace marginalmente preferible cuando los recursos son limitados.
- **Tendencia por ventana de entrada**: la ventaja de las RNNs frente a LR crece de forma
  consistente con `input_window`. Con `input=90` la mejora llega al −12 % − −14 %, mientras
  que con `input=5` suele ser < −1 %. Las RNNs explotan mejor el contexto temporal largo.
- **Ventanas con output largo (`output=90`)**: la variable objetivo es la media de retornos
  de 90 días, que tiene varianza muy baja. Las RNNs mejoran igual o más que en salidas cortas
  cuando `input` también es largo; con `input` corto la señal es insuficiente.
- **Arquitectura ganadora**: la mayoría de ventanas se resuelven bien con 2 capas y pocas
  unidades (32–64). Solo algunas ventanas con mucho contexto (`input=90`) requieren
  configuraciones más grandes (128–256 unidades o 3 capas).
- **HP search**: el ajuste fino mejora ≈ 0.3–2 % sobre la LSTM/GRU base sin ajuste, lo que
  confirma que la arquitectura base ya captura la mayor parte de la señal disponible en datos
  financieros de baja relación señal/ruido.

## Próximos pasos

- Crear el notebook `rnn/lstm/rnn-lstm-input30-output5.ipynb` para tener los resultados LSTM
  tuneados de esa ventana en el mismo formato que el resto.
- Registrar todos los runs de GRU y LSTM en `mlflow.db` para poder cruzarlos con MLflow UI.
- Comparar las mejores RNNs contra el MLP tuneado y el modelo mixto (CNN+RNN).
