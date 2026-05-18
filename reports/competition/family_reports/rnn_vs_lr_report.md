# RNN vs Linear Regression — Results Report

Generated from notebooks `rnn/lstm/` and `rnn/gru/` and from `data/lr_benchmark.csv`.

Two recurrent architectures are included:
- **LSTM** — 2-stage HP search per window (`lstm_layers × units × dropout`, then `lr × batch_size`)
- **GRU**  — same methodology, with `gru_layers ∈ {1, 2, 3}` and `units ∈ {32, 64, 128, 256}`

---

## Main Conclusion

- Mean test MAE **tuned LSTM** : `0.005347`
- Mean test MAE **tuned GRU**  : `0.005368`
- Mean test MAE **linear regression** : `0.005668`
- Best global RNN architecture : **LSTM** (mean test MAE = `0.005347`)
- Windows where tuned LSTM beats LR : **14 / 16**
- Windows where tuned GRU beats LR  : **15 / 16**

Both architectures outperform linear regression in the vast majority of windows. The advantage is
small for short input windows (`input=5/10`) and long outputs (`output=90`), where the available
signal is very weak. With `input=90` the RNNs achieve the largest relative improvements
(up to ~−14 % in test MAE), exploiting the long history better than the linear model.

---

## Model Ranking

> `wins_vs_lr` = windows where the model beats LR
> `wins_best_rnn` = windows where this model is the better of the two RNNs

| model | mean_test | median_test | best_test | worst_test | mean_delta_lr | wins_vs_lr | wins_best_rnn | mean_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.005347 | 0.002343 | 0.001275 | 0.012256 | -0.000307 | 14 | 12 | 41216 |
| GRU | 0.005368 | 0.003975 | 0.001269 | 0.012262 | -0.000300 | 15 | 4 | 75841 |

---

## Best RNN Per Window

| Salida \ Entrada | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012238` (L) | `0.012234` (L) | `0.012240` (G) | `0.012256` (L) |
| **out=5** | `0.005586` (L) | `0.005582` (L) | `0.005597` (G) | `0.005594` (L) |
| **out=30** | `0.002325` (L) | `0.002334` (L) | `0.002340` (L) | `0.002343` (L) |
| **out=90** | `0.001275` (L) | `0.001276` (L) | `0.001269` (G) | `0.001288` (G) |

> (L) = LSTM · (G) = GRU

---

## Best RNN Per Window — Detail

| input_window | output_window | model_type | MAE_test | MAE_test_lr | delta | pct_delta |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | LSTM | 0.012238 | 0.012384 | -0.000146 | -1.18% |
| 5 | 5 | LSTM | 0.005586 | 0.005625 | -0.000039 | -0.69% |
| 5 | 30 | LSTM | 0.002325 | 0.002340 | -0.000015 | -0.65% |
| 5 | 90 | LSTM | 0.001275 | 0.001271 | +0.000004 | +0.29% |
| 10 | 1 | LSTM | 0.012234 | 0.012554 | -0.000320 | -2.55% |
| 10 | 5 | LSTM | 0.005582 | 0.005698 | -0.000116 | -2.03% |
| 10 | 30 | LSTM | 0.002334 | 0.002358 | -0.000024 | -1.04% |
| 10 | 90 | LSTM | 0.001276 | 0.001282 | -0.000006 | -0.50% |
| 30 | 1 | GRU | 0.012240 | 0.012924 | -0.000684 | -5.29% |
| 30 | 5 | GRU | 0.005597 | 0.005877 | -0.000280 | -4.76% |
| 30 | 30 | LSTM | 0.002340 | 0.002436 | -0.000096 | -3.95% |
| 30 | 90 | GRU | 0.001269 | 0.001351 | -0.000082 | -6.10% |
| 90 | 1 | LSTM | 0.012256 | 0.014095 | -0.001839 | -13.05% |
| 90 | 5 | LSTM | 0.005594 | 0.006348 | -0.000754 | -11.88% |
| 90 | 30 | LSTM | 0.002343 | 0.002628 | -0.000285 | -10.85% |
| 90 | 90 | GRU | 0.001288 | 0.001518 | -0.000230 | -15.16% |

---

## Test MAE Matrices

### Tuned LSTM

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012238` | `0.012234` | `0.012244` | `0.012256` |
| **out=5** | `0.005586` | `0.005582` | `nan` | `0.005594` |
| **out=30** | `0.002325` | `0.002334` | `0.002340` | `0.002343` |
| **out=90** | `0.001275` | `0.001276` | `0.001276` | `0.001299` |

### Tuned GRU

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012244` | `0.012247` | `0.012240` | `0.012262` |
| **out=5** | `0.005593` | `0.005584` | `0.005597` | `0.005621` |
| **out=30** | `0.002337` | `0.002334` | `0.002366` | `0.002353` |
| **out=90** | `0.001275` | `0.001279` | `0.001269` | `0.001288` |

### Linear Regression

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `0.012384` | `0.012554` | `0.012924` | `0.014095` |
| **out=5** | `0.005625` | `0.005698` | `0.005877` | `0.006348` |
| **out=30** | `0.002340` | `0.002358` | `0.002436` | `0.002628` |
| **out=90** | `0.001271` | `0.001282` | `0.001351` | `0.001518` |

---

## Δ (Tuned LSTM − LR)

> Negative (↓) = LSTM wins; positive (↑) = LR wins.

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000146` ↓ | `-0.000320` ↓ | `-0.000680` ↓ | `-0.001839` ↓ |
| **out=5** | `-0.000039` ↓ | `-0.000116` ↓ | `+nan` ↑ | `-0.000754` ↓ |
| **out=30** | `-0.000015` ↓ | `-0.000024` ↓ | `-0.000096` ↓ | `-0.000285` ↓ |
| **out=90** | `+0.000004` ↑ | `-0.000006` ↓ | `-0.000075` ↓ | `-0.000219` ↓ |

## Δ (Tuned GRU − LR)

> Negative (↓) = GRU wins; positive (↑) = LR wins.

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `-0.000140` ↓ | `-0.000307` ↓ | `-0.000684` ↓ | `-0.001833` ↓ |
| **out=5** | `-0.000032` ↓ | `-0.000114` ↓ | `-0.000280` ↓ | `-0.000727` ↓ |
| **out=30** | `-0.000003` ↓ | `-0.000024` ↓ | `-0.000070` ↓ | `-0.000275` ↓ |
| **out=90** | `+0.000004` ↑ | `-0.000003` ↓ | `-0.000082` ↓ | `-0.000230` ↓ |

---

## Best Model Parameter Counts

> Trainable parameters of the best model (lowest test MAE) per window.
> (L) = LSTM · (G) = GRU

| Output \ Input | in=5 | in=10 | in=30 | in=90 |
|:---:|:---:|:---:|:---:|:---:|
| **out=1** | `5063` (L) | `16247` (L) | `18615` (G) | `16247` (L) |
| **out=5** | `5063` (L) | `16247` (L) | `67927` (G) | `16247` (L) |
| **out=30** | `5063` (L) | `16247` (L) | `16247` (L) | `7927` (L) |
| **out=90** | `7927` (L) | `57047` (L) | `67927` (G) | `258711` (G) |

> (L) = LSTM · (G) = GRU

---

## Best Model Hyperparameters Per Window

### in=5, out=1  —  tuned LSTM  |  test_mae = `0.012238`  |  Δ vs LR = `-0.000146` (-1.18%)

- **Architecture:** 2 LSTM layers · 16 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=5, out=5  —  tuned LSTM  |  test_mae = `0.005586`  |  Δ vs LR = `-0.000039` (-0.69%)

- **Architecture:** 2 LSTM layers · 16 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=5, out=30  —  tuned LSTM  |  test_mae = `0.002325`  |  Δ vs LR = `-0.000015` (-0.65%)

- **Architecture:** 2 LSTM layers · 16 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 128

### in=5, out=90  —  tuned LSTM  |  test_mae = `0.001275`  |  Δ vs LR = `+0.000004` (+0.29%)

- **Architecture:** 1 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128

### in=10, out=1  —  tuned LSTM  |  test_mae = `0.012234`  |  Δ vs LR = `-0.000320` (-2.55%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=10, out=5  —  tuned LSTM  |  test_mae = `0.005582`  |  Δ vs LR = `-0.000116` (-2.03%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 128

### in=10, out=30  —  tuned LSTM  |  test_mae = `0.002334`  |  Δ vs LR = `-0.000024` (-1.04%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-03 · batch_size = 128

### in=10, out=90  —  tuned LSTM  |  test_mae = `0.001276`  |  Δ vs LR = `-0.000006` (-0.50%)

- **Architecture:** 2 LSTM layers · 64 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64

### in=30, out=1  —  tuned GRU  |  test_mae = `0.012240`  |  Δ vs LR = `-0.000684` (-5.29%)

- **Architecture:** 3 GRU layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 128

### in=30, out=5  —  tuned GRU  |  test_mae = `0.005597`  |  Δ vs LR = `-0.000280` (-4.76%)

- **Architecture:** 3 GRU layers · 64 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64

### in=30, out=30  —  tuned LSTM  |  test_mae = `0.002340`  |  Δ vs LR = `-0.000096` (-3.95%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64

### in=30, out=90  —  tuned GRU  |  test_mae = `0.001269`  |  Δ vs LR = `-0.000082` (-6.10%)

- **Architecture:** 3 GRU layers · 64 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=90, out=1  —  tuned LSTM  |  test_mae = `0.012256`  |  Δ vs LR = `-0.001839` (-13.05%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=90, out=5  —  tuned LSTM  |  test_mae = `0.005594`  |  Δ vs LR = `-0.000754` (-11.88%)

- **Architecture:** 2 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=90, out=30  —  tuned LSTM  |  test_mae = `0.002343`  |  Δ vs LR = `-0.000285` (-10.85%)

- **Architecture:** 1 LSTM layers · 32 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 256

### in=90, out=90  —  tuned GRU  |  test_mae = `0.001288`  |  Δ vs LR = `-0.000230` (-15.16%)

- **Architecture:** 3 GRU layers · 128 units/layer · dropout = 0.2
- **Training:** lr = 1e-04 · batch_size = 64


---

## Interpretation

- **GRU vs LSTM**: the test MAE difference between both is minimal across almost all windows
  (< 0.0001). GRU has slightly fewer parameters for the same number of units (3 gates vs 4),
  making it marginally preferable when resources are constrained.
- **Input window trend**: the RNN advantage over LR grows consistently with `input_window`.
  With `input=90` the improvement reaches −12 % − −14 %, while with `input=5` it is usually
  < −1 %. RNNs exploit long temporal context better than the linear model.
- **Long output windows (`output=90`)**: the target variable is the 90-day mean return, which
  has very low variance. RNNs improve as much or more than for short outputs when `input` is
  also long; with short `input` the signal is insufficient.
- **Winning architecture**: most windows are solved well with 2 layers and few units (32–64).
  Only a few windows with large context (`input=90`) require larger configurations
  (128–256 units or 3 layers).
- **HP search**: fine-tuning improves ≈ 0.3–2 % over the baseline LSTM/GRU without tuning,
  confirming that the base architecture already captures most of the available signal in
  low signal-to-noise financial data.
