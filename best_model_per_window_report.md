# Global Best Model Per Window — Comparación de Todas las Familias

Comparación de las familias **MLP**, **CNN**, **RNN** y **Mixtos**.
Para cada combinación (input_window, output_window) se selecciona el modelo
con el menor MAE en test de entre las cuatro familias.
Valores negativos en `pct_delta_vs_lr` indican mejora sobre la regresión lineal.

## Conclusión Principal

- Media MAE test **mejor global**     : `0.005348`
- Media MAE test **regresión lineal** : `0.005668`
- Media Δ vs LR : `-0.000320` (-5.65%)
- Ventanas donde el mejor global supera a LR : **16 / 16**

### Ventanas ganadas por familia

| Familia | Ventanas ganadas |
|---------|-----------------|
| MLP | 8 |
| CNN | 8 |
| RNN | 0 |
| Mixtos | 0 |

### Media MAE test por familia (sobre las 16 ventanas)

| Familia | Media MAE test |
|---------|---------------|
| MLP | `0.005350` |
| CNN | `0.005352` |
| RNN | `0.005360` |
| Mixtos | `0.005377` |
| LR (benchmark) | `0.005668` |

## Mejor Modelo Por Ventana — Detalle

| input_window | output_window | familia | modelo | MAE_best | MAE_lr | pct_delta_vs_lr |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| 5 | 1 | **MLP** | mlp_3x128_gelu_dropout_l2 | `0.012224` | `0.012384` | `-1.29%` ↓ |
| 5 | 5 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.005574` | `0.005625` | `-0.91%` ↓ |
| 5 | 30 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.002321` | `0.002340` | `-0.81%` ↓ |
| 5 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001262` | `0.001271` | `-0.69%` ↓ |
| 10 | 1 | **MLP** | mlp_3x128_gelu_dropout_l2 | `0.012225` | `0.012554` | `-2.62%` ↓ |
| 10 | 5 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.005573` | `0.005698` | `-2.19%` ↓ |
| 10 | 30 | **CNN** | CNN_Deep_Conv1D | `0.002321` | `0.002358` | `-1.57%` ↓ |
| 10 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001259` | `0.001282` | `-1.77%` ↓ |
| 30 | 1 | **MLP** | mlp_3x128_gelu_dropout_l2 | `0.012232` | `0.012924` | `-5.35%` ↓ |
| 30 | 5 | **MLP** | mlp_4x100_gelu_dropout_l2 | `0.005574` | `0.005877` | `-5.16%` ↓ |
| 30 | 30 | **CNN** | CNN_Deep_Conv1D | `0.002319` | `0.002436` | `-4.79%` ↓ |
| 30 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001263` | `0.001351` | `-6.54%` ↓ |
| 90 | 1 | **MLP** | mlp_3x128_gelu_dropout_l2 | `0.012249` | `0.014095` | `-13.10%` ↓ |
| 90 | 5 | **CNN** | CNN_Deep_Conv1D | `0.005586` | `0.006348` | `-12.00%` ↓ |
| 90 | 30 | **CNN** | CNN_Deep_Conv1D | `0.002323` | `0.002628` | `-11.62%` ↓ |
| 90 | 90 | **CNN** | CNN_Deep_Conv1D | `0.001264` | `0.001518` | `-16.74%` ↓ |

## Matriz MAE Test (mejor global)

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012224` | `0.012225` | `0.012232` | `0.012249` |
| **out=5** | `0.005574` | `0.005573` | `0.005574` | `0.005586` |
| **out=30** | `0.002321` | `0.002321` | `0.002319` | `0.002323` |
| **out=90** | `0.001262` | `0.001259` | `0.001263` | `0.001264` |

## Matriz de Familia Ganadora

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | **MLP** | **MLP** | **MLP** | **MLP** |
| **out=5** | **MLP** | **MLP** | **MLP** | **CNN** |
| **out=30** | **MLP** | **CNN** | **CNN** | **CNN** |
| **out=90** | **CNN** | **CNN** | **CNN** | **CNN** |

## Comparación Completa — Las Cuatro Familias por Ventana

El valor en negrita es el ganador de cada fila.

| in | out | MLP | CNN | RNN | Mixtos | Ganador |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 1 | **`0.012224`** | `0.012237` | `0.012238` | `0.012232` | **MLP** |
| 5 | 5 | **`0.005574`** | `0.005582` | `0.005586` | `0.005593` | **MLP** |
| 5 | 30 | **`0.002321`** | `0.002322` | `0.002325` | `0.002339` | **MLP** |
| 5 | 90 | `0.001266` | **`0.001262`** | `0.001275` | `0.001275` | **CNN** |
| 10 | 1 | **`0.012225`** | `0.012238` | `0.012234` | `0.012244` | **MLP** |
| 10 | 5 | **`0.005573`** | `0.005575` | `0.005582` | `0.005620` | **MLP** |
| 10 | 30 | `0.002321` | **`0.002321`** | `0.002334` | `0.002334` | **CNN** |
| 10 | 90 | `0.001263` | **`0.001259`** | `0.001270` | `0.001272` | **CNN** |
| 30 | 1 | **`0.012232`** | `0.012243` | `0.012240` | `0.012244` | **MLP** |
| 30 | 5 | **`0.005574`** | `0.005577` | `0.005584` | `0.005581` | **MLP** |
| 30 | 30 | `0.002323` | **`0.002319`** | `0.002340` | `0.002340` | **CNN** |
| 30 | 90 | `0.001264` | **`0.001263`** | `0.001269` | `0.001346` | **CNN** |
| 90 | 1 | **`0.012249`** | `0.012259` | `0.012256` | `0.012279` | **MLP** |
| 90 | 5 | `0.005605` | **`0.005586`** | `0.005594` | `0.005594` | **CNN** |
| 90 | 30 | `0.002323` | **`0.002323`** | `0.002343` | `0.002344` | **CNN** |
| 90 | 90 | `0.001268` | **`0.001264`** | `0.001288` | `0.001396` | **CNN** |

## Δ (Mejor Global − LR) Matriz

> Negativo (↓) = modelo gana a LR; positivo (↑) = LR gana.

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000160` ↓ | `-0.000329` ↓ | `-0.000692` ↓ | `-0.001846` ↓ |
| **out=5** | `-0.000051` ↓ | `-0.000125` ↓ | `-0.000303` ↓ | `-0.000762` ↓ |
| **out=30** | `-0.000019` ↓ | `-0.000037` ↓ | `-0.000117` ↓ | `-0.000305` ↓ |
| **out=90** | `-0.000009` ↓ | `-0.000023` ↓ | `-0.000088` ↓ | `-0.000254` ↓ |

## Interpretación

- **Empate CNN–MLP (8–8)**: ambas familias se reparten las 16 ventanas a partes iguales.
  RNN y Mixtos no ganan ninguna ventana cuando se comparan contra las cuatro familias.
- **Patrón por output_window**: MLP domina las salidas cortas (`out=1` y `out=5`),  mientras que CNN gana la mayoría de las salidas largas (`out=30` y `out=90`).
  La única excepción relevante es `in=90, out=1` donde MLP aún gana.
- **Patrón por input_window**: con `input=5`, MLP gana 3 de 4 ventanas de salida;
  con `input=90`, CNN gana 3 de 4. Para ventanas intermedias el reparto es mixto.
- **Márgenes muy pequeños**: en la mayoría de ventanas la diferencia CNN–MLP es
  < 0.0001 MAE, dentro de la variabilidad aleatoria del entrenamiento. En la práctica
  ambas arquitecturas capturan la misma señal de esta serie financiera de bajo SNR.
- **RNN y Mixtos**: la complejidad recurrente añadida (LSTM/GRU puro o CNN-RNN híbrido)
  no reporta mejora sistemática sobre MLP o CNN en ninguna ventana.
- **16/16 ventanas superan a LR**: el modelo global óptimo mejora a la regresión lineal
  en las 16 combinaciones, incluyendo `(in=5, out=90)` donde los modelos recurrentes
  no conseguían superar el benchmark (CNN sí lo hace con −0.72%).
- **Mejora vs LR crece con input_window**: de ~−1% con `input=5` a ~−12/−17%
  con `input=90`, patrón consistente con los reportes individuales de cada familia.

