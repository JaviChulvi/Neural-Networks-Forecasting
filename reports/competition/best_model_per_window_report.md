# Global Best Model Per Window — Comparación de Todas las Familias

Comparación de las familias **MLP**, **CNN**, **RNN** y **Mixtos**.
Para cada combinación (input_window, output_window) se selecciona el modelo
con el menor MAE en test de entre las cuatro familias.
Valores negativos en `pct_delta_vs_lr` indican mejora sobre la regresión lineal.

## Conclusión Principal

- Media MAE test **mejor global**     : `0.005350`
- Media MAE test **regresión lineal** : `0.005668`
- Media Δ vs LR : `-0.000319` (-5.62%)
- Ventanas donde el mejor global supera a LR : **16 / 16**

### Ventanas ganadas por familia

| Familia | Ventanas ganadas |
|---------|-----------------|
| CNN | 10 |
| RNN | 3 |
| MLP | 2 |
| Mixtos | 1 |

### Media MAE test por familia (sobre las 16 ventanas)

| Familia | Media MAE test |
|---------|---------------|
| CNN | `0.005352` |
| RNN | `0.005360` |
| Mixtos | `0.005376` |
| MLP | `0.005385` |
| LR (benchmark) | `0.005668` |

## Mejor Modelo Por Ventana — Detalle

| input_window | output_window | familia | modelo | MAE_best | MAE_lr | pct_delta_vs_lr |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| 5 | 1 | **Mixtos** | CNN-MLP (tuned) | `0.012213` | `0.012384` | `-1.38%` ↓ |
| 5 | 5 | **CNN** | CNN_Deep_Conv1D | `0.005582` | `0.005625` | `-0.75%` ↓ |
| 5 | 30 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.002320` | `0.002340` | `-0.86%` ↓ |
| 5 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001262` | `0.001271` | `-0.72%` ↓ |
| 10 | 1 | **RNN** | LSTM | `0.012234` | `0.012554` | `-2.55%` ↓ |
| 10 | 5 | **CNN** | CNN_Deep_Conv1D | `0.005575` | `0.005698` | `-2.16%` ↓ |
| 10 | 30 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.002319` | `0.002358` | `-1.65%` ↓ |
| 10 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001259` | `0.001282` | `-1.80%` ↓ |
| 30 | 1 | **RNN** | GRU | `0.012240` | `0.012924` | `-5.29%` ↓ |
| 30 | 5 | **CNN** | CNN_Deep_Conv1D | `0.005577` | `0.005877` | `-5.11%` ↓ |
| 30 | 30 | **CNN** | CNN_Deep_Conv1D | `0.002319` | `0.002436` | `-4.81%` ↓ |
| 30 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001263` | `0.001351` | `-6.57%` ↓ |
| 90 | 1 | **RNN** | LSTM | `0.012256` | `0.014095` | `-13.05%` ↓ |
| 90 | 5 | **CNN** | CNN_Deep_Conv1D | `0.005586` | `0.006348` | `-12.00%` ↓ |
| 90 | 30 | **CNN** | CNN_Deep_Conv1D | `0.002323` | `0.002628` | `-11.63%` ↓ |
| 90 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001264` | `0.001518` | `-16.75%` ↓ |

## Matriz MAE Test (mejor global)

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012213` | `0.012234` | `0.012240` | `0.012256` |
| **out=5** | `0.005582` | `0.005575` | `0.005577` | `0.005586` |
| **out=30** | `0.002320` | `0.002319` | `0.002319` | `0.002323` |
| **out=90** | `0.001262` | `0.001259` | `0.001263` | `0.001264` |

## Matriz de Familia Ganadora

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | **Mixtos** | **RNN** | **RNN** | **RNN** |
| **out=5** | **CNN** | **CNN** | **CNN** | **CNN** |
| **out=30** | **MLP** | **MLP** | **CNN** | **CNN** |
| **out=90** | **CNN** | **CNN** | **CNN** | **CNN** |

## Comparación Completa — Las Cuatro Familias por Ventana

El valor en negrita es el ganador de cada fila.

| in | out | MLP | CNN | RNN | Mixtos | Ganador |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 1 | `0.012273` | `0.012237` | `0.012238` | **`0.012213`** | **Mixtos** |
| 5 | 5 | `0.005598` | **`0.005582`** | `0.005586` | `0.005604` | **CNN** |
| 5 | 30 | **`0.002320`** | `0.002322` | `0.002325` | `0.002339` | **MLP** |
| 5 | 90 | `0.001268` | **`0.001262`** | `0.001275` | `0.001267` | **CNN** |
| 10 | 1 | `0.012270` | `0.012238` | **`0.012234`** | `0.012244` | **RNN** |
| 10 | 5 | `0.005609` | **`0.005575`** | `0.005582` | `0.005604` | **CNN** |
| 10 | 30 | **`0.002319`** | `0.002321` | `0.002334` | `0.002334` | **MLP** |
| 10 | 90 | `0.001268` | **`0.001259`** | `0.001270` | `0.001276` | **CNN** |
| 30 | 1 | `0.012282` | `0.012243` | **`0.012240`** | `0.012244` | **RNN** |
| 30 | 5 | `0.005611` | **`0.005577`** | `0.005584` | `0.005581` | **CNN** |
| 30 | 30 | `0.002360` | **`0.002319`** | `0.002340` | `0.002340` | **CNN** |
| 30 | 90 | `0.001285` | **`0.001263`** | `0.001269` | `0.001346` | **CNN** |
| 90 | 1 | `0.012293` | `0.012259` | **`0.012256`** | `0.012273` | **RNN** |
| 90 | 5 | `0.005623` | **`0.005586`** | `0.005594` | `0.005594` | **CNN** |
| 90 | 30 | `0.002397` | **`0.002323`** | `0.002343` | `0.002358` | **CNN** |
| 90 | 90 | `0.001389` | **`0.001264`** | `0.001288` | `0.001396` | **CNN** |

## Δ (Mejor Global − LR) Matriz

> Negativo (↓) = modelo gana a LR; positivo (↑) = LR gana.

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000171` ↓ | `-0.000320` ↓ | `-0.000684` ↓ | `-0.001839` ↓ |
| **out=5** | `-0.000042` ↓ | `-0.000123` ↓ | `-0.000300` ↓ | `-0.000762` ↓ |
| **out=30** | `-0.000020` ↓ | `-0.000039` ↓ | `-0.000117` ↓ | `-0.000306` ↓ |
| **out=90** | `-0.000009` ↓ | `-0.000023` ↓ | `-0.000089` ↓ | `-0.000254` ↓ |

## Interpretación

- **Familia líder**: `CNN` gana 10 de 16 ventanas; `RNN` queda detrás con 3.
- **Lectura por ventana**: usa la matriz de familia ganadora para ver dónde cambia
  la arquitectura preferida. Las diferencias de MAE son pequeñas en muchas celdas,
  así que conviene interpretar los ganadores como ranking empírico, no como dominancia
  estadística fuerte sin repetir semillas.
- **RNN y Mixtos**: cuando no ganan una celda, no significa que sean inútiles;
  significa que, con los resultados guardados actuales, otra familia tiene menor MAE test.
- **16/16 ventanas superan a LR**: el modelo global óptimo mejora a la regresión lineal
  en las 16 combinaciones.
- **Mejora vs LR crece con input_window**: de ~−1% con `input=5` a ~−12/−17%
  con `input=90`, patrón consistente con los reportes individuales de cada familia.

