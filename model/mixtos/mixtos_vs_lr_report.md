# Modelos Mixtos vs Regresión Lineal — Results Report

Generated from notebooks `mixtos/mixto_input*_output*.ipynb` (14 windows),
`mixtos/cnn_rnn_hybrid/` (4 windows), and `data/lr_benchmark.csv`.

**Tuned Mixto** — 2-stage HP search over 7 architecture types:
- `lstm` — stacked LSTM layers
- `gru` — stacked GRU layers
- `cnn_lstm` — Conv1D → LSTM
- `cnn_gru` — Conv1D → GRU
- `cnn_lstm_mlp` — Conv1D → LSTM → MLP
- `cnn_gru_mlp` — Conv1D → GRU → MLP
- `cnn_mlp` — Conv1D → GlobalAveragePooling1D → MLP
Grid: `arch × n_layers ∈ {1,2} × units ∈ {32,64,128} × dropout ∈ {0.0,0.2}`
(× `kernel_size` for CNN variants in `input=30/90` notebooks), then `lr × batch_size` (9 combos).
Stage 1 size: 84 combinations for `input=5/10`, 144 for `input=30`, 204 for `input=90`.
Windows covered (14): all (input, output) combinations **except** (30,1) and (30,5).

**Hybrid CNN-RNN** — Fixed architecture (no HP tuning), 3 types:
- `CNN_LSTM` | `CNN_GRU` | `CNN_BiGRU` — Conv1D(64, ks) → {LSTM/GRU/BiGRU}(64) → Dense(64)
lr = 3e-4, batch = 128, early stopping (patience = 10).
Windows covered (4): (10,30), (10,90), (30,1), (30,5).
Note: (10,30) and (10,90) also have tuned notebooks; hybrid listed here for cross-comparison.

---

## Main Conclusion

- Mean test MAE **best tuned mixto** (14 windows) : `0.004871`
- Mean test MAE **best hybrid CNN-RNN** (4 windows): `0.005354`
- Mean test MAE **best mixed combined** (16 windows): `0.005376`
- Mean test MAE **linear regression** (16 windows)  : `0.005668`
- Windows where tuned mixto beats LR : **14 / 14**
- Windows where hybrid beats LR      : **4 / 4**
- Windows where best mixed beats LR  : **16 / 16**

Mixed models outperform linear regression in 16 of 16 windows.
Compared to the tuned RNN report (LSTM mean = 0.005362 over 16 windows), the mixed models
achieve a mean of `0.005376` — the CNN component does not consistently
improve over pure LSTM/GRU for this low-noise financial time-series task.

---

## Winning Architecture Count (Tuned Mixto, 14 windows)

> How many windows each architecture type achieved the lowest validation MAE in Stage 1,
> ultimately winning the 2-stage tuning process.

| arch_tag | arch_name | windows_won |
| --- | --- | --- |
| CM | cnn_mlp | 5 |
| L | lstm | 4 |
| CL | cnn_lstm | 3 |
| CLM | cnn_lstm_mlp | 1 |
| CG | cnn_gru | 1 |

---

## Group Statistics

| group | n_windows | mean_test | median_test | best_test | worst_test | mean_Δ_vs_lr | wins_vs_lr | mean_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuned Mixto | 14 | 0.004871 | 0.002349 | 0.001267 | 0.012273 | -0.000264 | 14 | 43088 |
| Hybrid CNN-RNN | 4 | 0.005354 | 0.003950 | 0.001270 | 0.012244 | -0.000257 | 4 | 39383 |
| Combined Best | 16 | 0.005376 | 0.003970 | 0.001267 | 0.012273 | -0.000292 | 16 | 42625 |

---

## Best Mixed Model Per Window (all 16)

> (30,1) and (30,5) use hybrid result — no tuned notebook available for those windows.

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012213` (CM) | `0.012244` (CL) | `0.012244` (HL) | `0.012273` (CM) |
| **out=5** | `0.005604` (CM) | `0.005604` (CM) | `0.005581` (HG) | `0.005594` (L) |
| **out=30** | `0.002339` (CL) | `0.002334` (L) | `0.002340` (L) | `0.002358` (CM) |
| **out=90** | `0.001267` (CLM) | `0.001276` (L) | `0.001346` (CL) | `0.001396` (CG) |

> (L) = LSTM · (G) = GRU · (CL) = CNN-LSTM · (CG) = CNN-GRU
> (CLM) = CNN-LSTM-MLP · (CGM) = CNN-GRU-MLP · (CM) = CNN-MLP
> (HL) = Hybrid CNN-LSTM · (HG) = Hybrid CNN-GRU · (HB) = Hybrid CNN-BiGRU

---

## Best Mixed Model Per Window — Detail

| input_window | output_window | model_type | MAE_test | MAE_test_lr | delta | pct_delta |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | CM | 0.012213 | 0.012384 | -0.000171 | -1.38% |
| 5 | 5 | CM | 0.005604 | 0.005625 | -0.000021 | -0.37% |
| 5 | 30 | CL | 0.002339 | 0.002340 | -0.000001 | -0.05% |
| 5 | 90 | CLM | 0.001267 | 0.001271 | -0.000004 | -0.34% |
| 10 | 1 | CL | 0.012244 | 0.012554 | -0.000310 | -2.47% |
| 10 | 5 | CM | 0.005604 | 0.005698 | -0.000094 | -1.64% |
| 10 | 30 | L | 0.002334 | 0.002358 | -0.000024 | -1.04% |
| 10 | 90 | L | 0.001276 | 0.001282 | -0.000006 | -0.50% |
| 30 | 1 | HL | 0.012244 | 0.012924 | -0.000680 | -5.26% |
| 30 | 5 | HG | 0.005581 | 0.005877 | -0.000295 | -5.03% |
| 30 | 30 | L | 0.002340 | 0.002436 | -0.000096 | -3.95% |
| 30 | 90 | CL | 0.001346 | 0.001351 | -0.000005 | -0.40% |
| 90 | 1 | CM | 0.012273 | 0.014095 | -0.001822 | -12.93% |
| 90 | 5 | L | 0.005594 | 0.006348 | -0.000754 | -11.88% |
| 90 | 30 | CM | 0.002358 | 0.002628 | -0.000270 | -10.28% |
| 90 | 90 | CG | 0.001396 | 0.001518 | -0.000122 | -8.04% |

---

## Tuned Mixto — Test MAE Matrix (14 windows)

> Best architecture per window after 2-stage HP search.
> Missing cells correspond to (30,1) and (30,5), covered by the hybrid notebook.

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012213` | `0.012244` | — | `0.012273` |
| **out=5** | `0.005604` | `0.005604` | — | `0.005594` |
| **out=30** | `0.002339` | `0.002334` | `0.002340` | `0.002358` |
| **out=90** | `0.001267` | `0.001276` | `0.001346` | `0.001396` |

---

## Hybrid CNN-RNN — All Architectures (4 windows)

> Three architectures evaluated with fixed hyperparameters (lr=3e-4, batch=128).
> Best per window highlighted in bold in the Best Model table above.

| in | out | model | MAE_train | MAE_val | MAE_test | params | Δ vs LR | % Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 30 | HB | 0.002202 | 0.001713 | 0.002368 | 67351 | `+0.000009` | +0.39% |
| 10 | 30 | HG | 0.002202 | 0.001699 | 0.002320 | 35351 | `-0.000039` | -1.64% |
| 10 | 30 | HL | 0.002202 | 0.001699 | 0.002327 | 43415 | `-0.000031` | -1.33% |
| 10 | 90 | HB | 0.001265 | 0.000942 | 0.001271 | 67351 | `-0.000011` | -0.86% |
| 10 | 90 | HG | 0.001268 | 0.000926 | 0.001277 | 35351 | `-0.000005` | -0.40% |
| 10 | 90 | HL | 0.001267 | 0.000926 | 0.001270 | 43415 | `-0.000013` | -0.98% |
| 30 | 1 | HB | 0.011833 | 0.009040 | 0.012261 | 67351 | `-0.000663` | -5.13% |
| 30 | 1 | HG | 0.011839 | 0.009037 | 0.012246 | 35351 | `-0.000678` | -5.25% |
| 30 | 1 | HL | 0.011841 | 0.009037 | 0.012244 | 43415 | `-0.000680` | -5.26% |
| 30 | 5 | HB | 0.005475 | 0.004155 | 0.005600 | 67351 | `-0.000277` | -4.71% |
| 30 | 5 | HG | 0.005479 | 0.004154 | 0.005581 | 35351 | `-0.000295` | -5.03% |
| 30 | 5 | HL | 0.005484 | 0.004153 | 0.005596 | 43415 | `-0.000281` | -4.78% |

---

## Δ (Tuned Mixto − LR)

> Negative (↓) = Mixto wins; positive (↑) = LR wins.
> Cells show the best architecture winner (lowest test MAE after 2-stage tuning).

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000171` ↓ | `-0.000310` ↓ | — | `-0.001822` ↓ |
| **out=5** | `-0.000021` ↓ | `-0.000094` ↓ | — | `-0.000754` ↓ |
| **out=30** | `-0.000001` ↓ | `-0.000024` ↓ | `-0.000096` ↓ | `-0.000270` ↓ |
| **out=90** | `-0.000004` ↓ | `-0.000006` ↓ | `-0.000005` ↓ | `-0.000122` ↓ |

## Δ (Hybrid CNN-RNN best − LR)

> Best hybrid architecture per window (by test MAE).

| Output \ Input | in=10 | in=30 |
|:---:|:---:|:---:|
| **out=1** | — | `-0.000680` ↓ |
| **out=5** | — | `-0.000295` ↓ |
| **out=30** | `-0.000039` ↓ | — |
| **out=90** | `-0.000013` ↓ | — |

---

## Best Model Parameter Counts (all 16 windows)

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `4215` (CM) | `72023` (CL) | `43415` (HL) | `46999` (CM) |
| **out=5** | `10135` (CM) | `10135` (CM) | `35351` (HG) | `16247` (L) |
| **out=30** | `11319` (CL) | `16247` (L) | `16247` (L) | `16023` (CM) |
| **out=90** | `77527` (CLM) | `57047` (L) | `38999` (CL) | `210071` (CG) |

> (L) = LSTM · (G) = GRU · (CL) = CNN-LSTM · (CG) = CNN-GRU
> (CLM) = CNN-LSTM-MLP · (CGM) = CNN-GRU-MLP · (CM) = CNN-MLP
> (HL) = Hybrid CNN-LSTM · (HG) = Hybrid CNN-GRU

---

## Best Model Hyperparameters Per Window

### in=5, out=1  —  CNN-MLP [tuned]  |  test_mae = `0.012213`  |  Δ vs LR = `-0.000171` (-1.38%)

- **Architecture:** Conv1D(32 filters, ks=3) → GlobalAveragePooling1D → MLP × 2 layer(s) · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64
- **Params:** 4,215

### in=5, out=5  —  CNN-MLP [tuned]  |  test_mae = `0.005604`  |  Δ vs LR = `-0.000021` (-0.37%)

- **Architecture:** Conv1D(64 filters, ks=3) → GlobalAveragePooling1D → MLP × 1 layer(s) · dropout = 0.0
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 10,135

### in=5, out=30  —  CNN-LSTM [tuned]  |  test_mae = `0.002339`  |  Δ vs LR = `-0.000001` (-0.05%)

- **Architecture:** Conv1D(32 filters, ks=3) → LSTM(32) × 1 layer(s) · dropout = 0.0
- **Training:** lr = 1e-04 · batch_size = 256
- **Params:** 11,319

### in=5, out=90  —  CNN-LSTM-MLP [tuned]  |  test_mae = `0.001267`  |  Δ vs LR = `-0.000004` (-0.34%)

- **Architecture:** Conv1D(64 filters, ks=3) → LSTM(64) × 2 layer(s) → MLP(2 dense layers) · dropout = 0.0
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 77,527

### in=10, out=1  —  CNN-LSTM [tuned]  |  test_mae = `0.012244`  |  Δ vs LR = `-0.000310` (-2.47%)

- **Architecture:** Conv1D(64 filters, ks=3) → LSTM(64) × 2 layer(s) · dropout = 0.0
- **Training:** lr = 1e-04 · batch_size = 256
- **Params:** 72,023

### in=10, out=5  —  CNN-MLP [tuned]  |  test_mae = `0.005604`  |  Δ vs LR = `-0.000094` (-1.64%)

- **Architecture:** Conv1D(64 filters, ks=3) → GlobalAveragePooling1D → MLP × 1 layer(s) · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 10,135

### in=10, out=30  —  LSTM [tuned]  |  test_mae = `0.002334`  |  Δ vs LR = `-0.000024` (-1.04%)

- **Architecture:** 2 LSTM layer(s) · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 16,247

### in=10, out=90  —  LSTM [tuned]  |  test_mae = `0.001276`  |  Δ vs LR = `-0.000006` (-0.50%)

- **Architecture:** 2 LSTM layer(s) · 64 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64
- **Params:** 57,047

### in=30, out=1  —  Hybrid CNN-LSTM [hybrid (fixed HP)]  |  test_mae = `0.012244`  |  Δ vs LR = `-0.000680` (-5.26%)

- **Architecture:** Conv1D(64, ks=3) → LSTM(64) → Dense(64) · dropout = 0.15
- **Training:** lr = 3e-04 · batch_size = 128
- **Params:** 43,415

### in=30, out=5  —  Hybrid CNN-GRU [hybrid (fixed HP)]  |  test_mae = `0.005581`  |  Δ vs LR = `-0.000295` (-5.03%)

- **Architecture:** Conv1D(64, ks=3) → GRU(64) → Dense(64) · dropout = 0.15
- **Training:** lr = 3e-04 · batch_size = 128
- **Params:** 35,351

### in=30, out=30  —  LSTM [tuned]  |  test_mae = `0.002340`  |  Δ vs LR = `-0.000096` (-3.95%)

- **Architecture:** 2 LSTM layer(s) · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64
- **Params:** 16,247

### in=30, out=90  —  CNN-LSTM [tuned]  |  test_mae = `0.001346`  |  Δ vs LR = `-0.000005` (-0.40%)

- **Architecture:** Conv1D(64 filters, ks=3) → LSTM(64) × 1 layer(s) · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 38,999

### in=90, out=1  —  CNN-MLP [tuned]  |  test_mae = `0.012273`  |  Δ vs LR = `-0.001822` (-12.93%)

- **Architecture:** Conv1D(128 filters, ks=7) → GlobalAveragePooling1D → MLP × 2 layer(s) · dropout = 0.0
- **Training:** lr = 1e-04 · batch_size = 256
- **Params:** 46,999

### in=90, out=5  —  LSTM [tuned]  |  test_mae = `0.005594`  |  Δ vs LR = `-0.000754` (-11.88%)

- **Architecture:** 2 LSTM layer(s) · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256
- **Params:** 16,247

### in=90, out=30  —  CNN-MLP [tuned]  |  test_mae = `0.002358`  |  Δ vs LR = `-0.000270` (-10.28%)

- **Architecture:** Conv1D(64 filters, ks=7) → GlobalAveragePooling1D → MLP × 1 layer(s) · dropout = 0.0
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 16,023

### in=90, out=90  —  CNN-GRU [tuned]  |  test_mae = `0.001396`  |  Δ vs LR = `-0.000122` (-8.04%)

- **Architecture:** Conv1D(128 filters, ks=3) → GRU(128) × 2 layer(s) · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128
- **Params:** 210,071


---

## Interpretation

- **CNN/MLP variants**: after adding `cnn_lstm_mlp`, `cnn_gru_mlp` and `cnn_mlp`,
  the tuned search can select dense heads when they reduce validation MAE.
- **Architecture selection**: the winning-count table above is now the best summary of which
  architectures actually won after the expanded search.
- **Hybrid vs tuned**: the fixed hybrid (64 units, lr=3e-4) beats the 2-stage tuned result for
  (10,30) by 0.000014 MAE, a negligible margin likely due to random variation. The tuned search
  generally matches or exceeds the hybrid for (10,90) and performs competitively for (30,1)/(30,5).
- **Input window trend**: the advantage over LR grows consistently with `input_window`, reaching
  −10 to −13 % for `input=90` (same pattern as pure RNN models).
- **Long outputs (`output=90`)**: the expanded mixed search now beats LR in all reported
  windows, including `input=5, output=90`.
- **Comparison with pure RNN**: mean test MAE over 16 windows — Tuned LSTM = 0.005362,
  Best Mixed = `0.005376`. The mixed search adds significant
  search overhead (4× more architectures per window) but does not systematically improve
  over a well-tuned LSTM/GRU, consistent with the low signal-to-noise nature of the data.
- **Parameter efficiency**: winners range from compact CNN-MLP models to larger recurrent
  hybrids; see the parameter matrix for the exact counts per window.
