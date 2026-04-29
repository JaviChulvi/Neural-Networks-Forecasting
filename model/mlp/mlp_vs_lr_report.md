# MLP vs Linear Regression Report

Generated: 2026-04-29

This report compares all MLP models in `model/mlp/` against the linear
regression benchmark stored in `data/lr_benchmark.csv`.

## Executive Summary

The best MLP architecture is clearly `mlp_2x100_dropout`.

It is the best MLP in 15 of the 16 input/output window combinations, has the
lowest average test MAE among the neural models, and is the only MLP that
competes seriously with the linear regression benchmark across the whole grid.

However, the MLP family does not consistently beat linear regression yet.

- LR wins 10 of 16 window combinations when compared against the best available
  MLP for each cell.
- The best MLP wins 6 of 16 cells, all with input windows of 30 or 90 days.
- For short input windows of 5 and 10 days, LR wins every output horizon.
- For input window 90, `mlp_2x100_dropout` wins every output horizon.
- The strongest MLP advantage appears when the model receives a long historical
  window. The best result is `input_window=90`, `output_window=1`, where
  dropout MLP improves LR by about 9.0%.
- The best average MLP, `mlp_2x100_dropout`, has mean test MAE `0.005618`
  versus LR mean test MAE `0.005668`. That is a tiny absolute average advantage,
  but it is not uniform: short-window cells are still worse than LR.

Main conclusion:

The current dense MLP approach can beat LR in selected regimes, especially with
long input histories and dropout regularization. But as currently configured,
plain dense MLPs are not reliably better than LR. To make this kind of model
more convincing, the next iteration should use early stopping with restored best
weights, multiple random seeds, time-series cross-validation, stronger
regularization strategy, and probably a residual/wide-and-deep design that lets
the neural network learn corrections on top of the LR baseline.

## Experiment Setup

All models predict future average log returns for 23 S&P 500 assets:

`AEP`, `BA`, `CAT`, `CNP`, `CVX`, `DIS`, `DTE`, `ED`, `GD`, `GE`, `HON`,
`HPQ`, `IBM`, `IP`, `JNJ`, `KO`, `KR`, `MMM`, `MO`, `MRK`, `MSI`, `PG`, `XOM`.

Data:

- Source file: `data/returns.parquet`
- Shape: `16185` rows x `23` assets
- Date range: `1962-01-03` to `2026-04-24`
- Test split: final 10% of samples, with `shuffle=False`
- Validation split for MLPs: final 10% of the training fold, with
  `shuffle=False`

Window grid:

| Input window | Output windows |
| ------------ | -------------- |
| 5 days       | 1, 5, 30, 90   |
| 10 days      | 1, 5, 30, 90   |
| 30 days      | 1, 5, 30, 90   |
| 90 days      | 1, 5, 30, 90   |

For each sample:

- `X` has shape `(samples, input_window, 23)`.
- Dense models flatten `X` to `(samples, input_window * 23)`.
- `y` has shape `(samples, 23)`.
- `y` is the average future return over the output window.
- Metric is MAE averaged across samples and assets.

## Models Compared

All MLPs use:

- Keras `Sequential`
- Adam optimizer
- learning rate `1e-3`
- batch size `128`
- loss: `mean_absolute_error`
- seed: `RANDOM_SEED = 42`
- output layer: `Dense(23)` with linear activation

| Model | Architecture | Regularization | Epochs | Parameter range | Design intent |
| ----- | ------------ | -------------- | ------ | --------------- | ------------- |
| `mlp_1x64_relu` | Flatten input -> Dense(64, ReLU) -> Dense(23) | None | 500 | 8,919 to 134,039 | Small baseline neural net; low capacity. |
| `mlp_1x100_relu` | Flatten input -> Dense(100, ReLU) -> Dense(23) | None | 500 | 13,923 to 209,423 | Slightly wider shallow MLP. |
| `mlp_2x100_relu` | Flatten input -> Dense(100, ReLU) -> Dense(100, ReLU) -> Dense(23) | None | 500 | 24,023 to 219,523 | Deeper nonlinear model with same hidden width. |
| `mlp_3x200_relu` | Flatten input -> Dense(200, ReLU) x3 -> Dense(23) | None | 200 | 108,223 to 499,223 | High-capacity dense model. |
| `mlp_2x100_dropout` | Flatten input -> Dense(100, ReLU) -> Dropout(0.2) -> Dense(100, ReLU) -> Dropout(0.2) -> Dense(23) | Dropout 0.2 | 200 | 24,023 to 219,523 | Same capacity as 2x100, but regularized. |
| `mlp_2x100_bn_l2` | Flatten input -> Dense(100, ReLU, L2=1e-4) -> BatchNorm -> Dense(100, ReLU, L2=1e-4) -> BatchNorm -> Dense(23) | BatchNorm + L2 | 200 | 24,823 to 220,323 | Regularized variant using weight decay and batch normalization. |

Parameter count grows with input window because flattening turns a longer
history into more input features:

| model_name        | 5      | 10     | 30     | 90     |
| ----------------- | ------ | ------ | ------ | ------ |
| mlp_1x100_relu    | 13923  | 25423  | 71423  | 209423 |
| mlp_1x64_relu     | 8919   | 16279  | 45719  | 134039 |
| mlp_2x100_bn_l2   | 24823  | 36323  | 82323  | 220323 |
| mlp_2x100_dropout | 24023  | 35523  | 81523  | 219523 |
| mlp_2x100_relu    | 24023  | 35523  | 81523  | 219523 |
| mlp_3x200_relu    | 108223 | 131223 | 223223 | 499223 |

## High-Level Performance

`delta_vs_lr = MLP MAE_test - LR MAE_test`.

- Negative delta means the MLP beats LR.
- Positive delta means LR is better.

| model_name        | mean_test | median_test | best_test | worst_test | mean_delta | mean_pct   | median_pct | wins | avg_params    | min_params | max_params | epochs | mean_train | mean_val | test_train_gap | test_val_gap |
| ----------------- | --------- | ----------- | --------- | ---------- | ---------- | ---------- | ---------- | ---- | ------------- | ---------- | ---------- | ------ | ---------- | -------- | -------------- | ------------ |
| mlp_2x100_dropout | 0.005618  | 0.004121    | 0.001387  | 0.012971   | -0.000050  | 0.542768   | 2.645855   | 6    | 90148.000000  | 24023      | 219523     | 200    | 0.004309   | 0.003995 | 0.001309       | 0.001623     |
| mlp_3x200_relu    | 0.006165  | 0.004479    | 0.001373  | 0.014438   | 0.000496   | 8.587228   | 10.778974  | 3    | 240473.000000 | 108223     | 499223     | 200    | 0.003786   | 0.003992 | 0.002379       | 0.002173     |
| mlp_2x100_relu    | 0.006678  | 0.004705    | 0.001425  | 0.015833   | 0.001009   | 17.725441  | 20.624203  | 1    | 90148.000000  | 24023      | 219523     | 500    | 0.004028   | 0.003998 | 0.002649       | 0.002680     |
| mlp_1x64_relu     | 0.006862  | 0.004597    | 0.001483  | 0.016718   | 0.001194   | 18.838316  | 20.467028  | 1    | 51239.000000  | 8919       | 134039     | 500    | 0.004352   | 0.004002 | 0.002509       | 0.002860     |
| mlp_1x100_relu    | 0.006970  | 0.004950    | 0.001611  | 0.017497   | 0.001302   | 22.490912  | 24.623668  | 0    | 80048.000000  | 13923      | 209423     | 500    | 0.004109   | 0.004004 | 0.002861       | 0.002967     |
| mlp_2x100_bn_l2   | 0.019015  | 0.008115    | 0.002770  | 0.140086   | 0.013346   | 207.713284 | 94.102003  | 1    | 90948.000000  | 24823      | 220323     | 200    | 0.018842   | 0.005333 | 0.000173       | 0.013682     |

Important observations:

1. `mlp_2x100_dropout` is the only model with a lower average absolute MAE
   than LR. The margin is tiny: `0.005618` vs `0.005668`.
2. The same model still has positive average percentage delta because percentage
   errors are sensitive to the small-MAE long-horizon cells. This is why the
   cell-level table matters more than the global mean.
3. Unregularized 500-epoch MLPs overfit or drift enough that additional epochs
   do not help. More training is not the answer by itself.
4. `mlp_3x200_relu` has strong capacity and wins some long-history cells, but it
   is worse than dropout on almost every cell.
5. `mlp_2x100_bn_l2` is not viable in the current setup. It has one isolated win
   but huge failures, especially for short input windows.

## LR Matrix vs Best MLP Matrix

LR test MAE:

| input_window | 1        | 5        | 30       | 90       |
| ------------ | -------- | -------- | -------- | -------- |
| 5            | 0.012384 | 0.005625 | 0.002340 | 0.001271 |
| 10           | 0.012554 | 0.005698 | 0.002358 | 0.001282 |
| 30           | 0.012924 | 0.005877 | 0.002436 | 0.001351 |
| 90           | 0.014095 | 0.006348 | 0.002628 | 0.001518 |

Best MLP test MAE per cell:

| input_window | 1        | 5        | 30       | 90       |
| ------------ | -------- | -------- | -------- | -------- |
| 5            | 0.012710 | 0.005774 | 0.002417 | 0.001414 |
| 10           | 0.012971 | 0.005852 | 0.002488 | 0.001412 |
| 30           | 0.012797 | 0.005754 | 0.002463 | 0.001373 |
| 90           | 0.012829 | 0.005806 | 0.002427 | 0.001387 |

Best MLP percentage delta vs LR:

| input_window | 1     | 5     | 30    | 90    |
| ------------ | ----- | ----- | ----- | ----- |
| 5            | 2.63  | 2.66  | 3.27  | 11.22 |
| 10           | 3.32  | 2.71  | 5.49  | 10.12 |
| 30           | -0.99 | -2.10 | 1.08  | 1.58  |
| 90           | -8.99 | -8.55 | -7.67 | -8.65 |

Winner including LR:

| input_window | 1                 | 5                 | 30                | 90                |
| ------------ | ----------------- | ----------------- | ----------------- | ----------------- |
| 5            | LR                | LR                | LR                | LR                |
| 10           | LR                | LR                | LR                | LR                |
| 30           | mlp_2x100_dropout | mlp_2x100_dropout | LR                | LR                |
| 90           | mlp_2x100_dropout | mlp_2x100_dropout | mlp_2x100_dropout | mlp_2x100_dropout |

This is the core result of the experiment.

Short input windows do not provide enough nonlinear signal for the MLPs to
justify their additional complexity. Long input windows are where the neural
model starts to exploit something LR does not capture as well.

## Best MLP by Window Pair

| input_window | output_window | model_name        | MAE_test | MAE_test_LR | delta_vs_lr | best_mlp_vs_lr_pct | winner_including_lr |
| ------------ | ------------- | ----------------- | -------- | ----------- | ----------- | ------------------ | ------------------- |
| 5            | 1             | mlp_2x100_dropout | 0.012710 | 0.012384    | 0.000326    | 2.631836           | LR                  |
| 5            | 5             | mlp_2x100_dropout | 0.005774 | 0.005625    | 0.000150    | 2.659874           | LR                  |
| 5            | 30            | mlp_2x100_dropout | 0.002417 | 0.002340    | 0.000077    | 3.273956           | LR                  |
| 5            | 90            | mlp_2x100_dropout | 0.001414 | 0.001271    | 0.000143    | 11.220080          | LR                  |
| 10           | 1             | mlp_2x100_dropout | 0.012971 | 0.012554    | 0.000417    | 3.320112           | LR                  |
| 10           | 5             | mlp_2x100_dropout | 0.005852 | 0.005698    | 0.000155    | 2.713440           | LR                  |
| 10           | 30            | mlp_2x100_dropout | 0.002488 | 0.002358    | 0.000130    | 5.493194           | LR                  |
| 10           | 90            | mlp_2x100_dropout | 0.001412 | 0.001282    | 0.000130    | 10.120314          | LR                  |
| 30           | 1             | mlp_2x100_dropout | 0.012797 | 0.012924    | -0.000127   | -0.985009          | mlp_2x100_dropout   |
| 30           | 5             | mlp_2x100_dropout | 0.005754 | 0.005877    | -0.000123   | -2.095626          | mlp_2x100_dropout   |
| 30           | 30            | mlp_2x100_dropout | 0.002463 | 0.002436    | 0.000026    | 1.078075           | LR                  |
| 30           | 90            | mlp_3x200_relu    | 0.001373 | 0.001351    | 0.000021    | 1.577623           | LR                  |
| 90           | 1             | mlp_2x100_dropout | 0.012829 | 0.014095    | -0.001267   | -8.987506          | mlp_2x100_dropout   |
| 90           | 5             | mlp_2x100_dropout | 0.005806 | 0.006348    | -0.000543   | -8.546252          | mlp_2x100_dropout   |
| 90           | 30            | mlp_2x100_dropout | 0.002427 | 0.002628    | -0.000202   | -7.672891          | mlp_2x100_dropout   |
| 90           | 90            | mlp_2x100_dropout | 0.001387 | 0.001518    | -0.000131   | -8.652019          | mlp_2x100_dropout   |

## MLP-Only Ranking

| model_name        | mean_rank | best_cells | second_or_better |
| ----------------- | --------- | ---------- | ---------------- |
| mlp_2x100_dropout | 1.06      | 15         | 16               |
| mlp_3x200_relu    | 2.12      | 1          | 13               |
| mlp_1x64_relu     | 3.81      | 0          | 1                |
| mlp_2x100_relu    | 3.81      | 0          | 1                |
| mlp_1x100_relu    | 4.75      | 0          | 0                |
| mlp_2x100_bn_l2   | 5.44      | 0          | 1                |

The ranking is extremely decisive:

- Dropout MLP is almost always best.
- Large 3x200 MLP is usually second.
- Shallow unregularized models are not competitive.
- BatchNorm + L2 is unstable or badly matched to this problem as configured.

## Model-by-Model Findings

### `mlp_1x64_relu`

Architecture:

```text
Flatten(input_window x 23)
Dense(64, relu)
Dense(23)
```

This is the smallest neural model. It is useful as a sanity check because it
tests whether a small nonlinear layer can improve the linear baseline.

Result:

- Mean test MAE: `0.006862`
- LR mean test MAE: `0.005668`
- Wins vs LR: `1 / 16`
- Only win: `input_window=90`, `output_window=90`, with about `2.28%`
  improvement.

Interpretation:

The model has too little capacity for long inputs and still overfits enough to
lose most cells. Its best result appears only in the longest input and output
configuration, where the target is smoother and easier to predict.

This model is not strong enough to justify replacing LR.

### `mlp_1x100_relu`

Architecture:

```text
Flatten(input_window x 23)
Dense(100, relu)
Dense(23)
```

This is a wider version of the shallow MLP.

Result:

- Mean test MAE: `0.006970`
- Wins vs LR: `0 / 16`
- Worst than `mlp_1x64_relu` on average despite having more units.

Interpretation:

Adding width without regularization does not help. The model has more capacity
but no clear mechanism to control overfitting or extract temporal structure.

The result suggests that shallow dense nonlinearity alone is not the missing
ingredient.

### `mlp_2x100_relu`

Architecture:

```text
Flatten(input_window x 23)
Dense(100, relu)
Dense(100, relu)
Dense(23)
```

This model tests whether adding depth improves feature interaction.

Result:

- Mean test MAE: `0.006678`
- Wins vs LR: `1 / 16`
- Only win: `input_window=90`, `output_window=90`, with about `6.16%`
  improvement.

Interpretation:

Depth helps slightly compared to shallow MLPs, but not enough. The model is
better than the shallow 100-unit MLP, but it still loses most cells. With 500
epochs and no regularization, it appears to learn training patterns that do not
transfer to the final test period.

This model is promising only for long-window settings, but the dropout version
is strictly more useful.

### `mlp_3x200_relu`

Architecture:

```text
Flatten(input_window x 23)
Dense(200, relu)
Dense(200, relu)
Dense(200, relu)
Dense(23)
```

This is the highest-capacity plain MLP.

Result:

- Mean test MAE: `0.006165`
- Wins vs LR: `3 / 16`
- Best MLP in only `1 / 16` cells: `input_window=30`, `output_window=90`
- Strongest cells are long input windows:
  - `90 -> 90`: `3.75%` better than LR
  - `90 -> 30`: `3.31%` better than LR
  - `90 -> 5`: `0.72%` better than LR

Interpretation:

The larger model does extract useful nonlinear signal in some long-history
cases, but it is less robust than dropout. It also has the largest parameter
count, up to nearly 500k parameters for `input_window=90`, and that extra
capacity does not translate into broad wins.

This model is useful evidence that MLP capacity can matter, but capacity alone
is not enough. Without regularization, the extra degrees of freedom mostly
increase variance.

### `mlp_2x100_dropout`

Architecture:

```text
Flatten(input_window x 23)
Dense(100, relu)
Dropout(0.2)
Dense(100, relu)
Dropout(0.2)
Dense(23)
```

This is the same hidden structure as `mlp_2x100_relu`, with dropout after each
hidden layer and only 200 epochs.

Result:

- Mean test MAE: `0.005618`
- LR mean test MAE: `0.005668`
- Wins vs LR: `6 / 16`
- Best MLP in `15 / 16` cells
- Second-or-better MLP in `16 / 16` cells
- Best improvement:
  - `input_window=90`, `output_window=1`: `8.99%` better than LR
  - `input_window=90`, `output_window=90`: `8.65%` better than LR
  - `input_window=90`, `output_window=5`: `8.55%` better than LR

Interpretation:

This is the most important model in the experiment.

Dropout changes the behavior materially. Compared with the unregularized
2x100 MLP, it has the same number of trainable parameters, fewer epochs, and
much better test behavior. That means the main issue was not capacity; it was
generalization.

The model is especially strong with `input_window=90`. This suggests that the
MLP is finding nonlinear relationships across longer lag structures and
cross-asset interactions that LR does not fully exploit.

Still, it loses all `input_window=5` and `input_window=10` cells. For short
histories, the nonlinear model has no clear advantage.

### `mlp_2x100_bn_l2`

Architecture:

```text
Flatten(input_window x 23)
Dense(100, relu, L2=1e-4)
BatchNormalization()
Dense(100, relu, L2=1e-4)
BatchNormalization()
Dense(23)
```

Result:

- Mean test MAE: `0.019015`
- Wins vs LR: `1 / 16`
- Severe failures:
  - `5 -> 1`: more than `1000%` worse than LR
  - `5 -> 5`: more than `900%` worse than LR
  - `10 -> 90`: about `297%` worse than LR

Interpretation:

This architecture should not be used as-is.

The validation loss shown in the result CSV can look much better than the final
test behavior, especially because the notebooks record the minimum validation
loss but predict test values using the final epoch weights. The model also
shows a large gap between validation and test behavior. Batch normalization can
be sensitive in time-series settings because the running statistics used at
inference may not match the final test distribution, especially under
non-stationarity.

If this variant is revisited, batch normalization should be tested carefully:

- Try `Dense -> BatchNorm -> Activation` instead of `Dense(relu) -> BatchNorm`.
- Use `EarlyStopping(restore_best_weights=True)`.
- Reduce learning rate.
- Tune L2 strength.
- Compare against L2 without BatchNorm.
- Compare against LayerNorm, which does not depend on moving batch statistics.

## Training Dynamics

The notebooks store `MAE_val` as the minimum validation loss:

```python
"MAE_val": min(hist.history["val_loss"])
```

But the test predictions are generated from the final epoch model:

```python
y_pred_test = model.predict(X_test, verbose=0)
```

This means `MAE_val` is optimistic if validation loss worsens after the best
epoch. The model is not automatically restored to the best validation epoch.

Training dynamics summary:

| model_name        | mean_best_epoch | median_best_epoch | min_best_epoch | max_best_epoch | mean_final_over_best_pct | median_final_over_best_pct | cells_final_gt_best_5pct | mean_final_train_loss | mean_final_val_loss |
| ----------------- | --------------- | ----------------- | -------------- | -------------- | ------------------------ | -------------------------- | ------------------------ | --------------------- | ------------------- |
| mlp_2x100_dropout | 27.375          | 15.500            | 1              | 106            | 3.948                    | 3.884                      | 6                        | 0.004                 | 0.004               |
| mlp_1x64_relu     | 14.125          | 3.500             | 1              | 127            | 8.254                    | 6.263                      | 12                       | 0.004                 | 0.004               |
| mlp_1x100_relu    | 8.688           | 7.500             | 1              | 27             | 10.544                   | 10.183                     | 15                       | 0.004                 | 0.004               |
| mlp_3x200_relu    | 22.750          | 3.000             | 1              | 185            | 13.970                   | 15.566                     | 15                       | 0.003                 | 0.005               |
| mlp_2x100_relu    | 9.062           | 4.000             | 1              | 56             | 19.903                   | 19.792                     | 16                       | 0.004                 | 0.005               |
| mlp_2x100_bn_l2   | 155.375         | 174.000           | 42             | 200            | 164.205                  | 47.908                     | 15                       | 0.005                 | 0.018               |

Interpretation:

- Dropout has the healthiest training behavior. Final validation loss is only
  about `3.95%` worse than the best validation loss on average.
- Plain MLPs often peak very early and then drift.
- `mlp_2x100_relu` is especially overtrained: all 16 cells end more than 5%
  above their best validation loss.
- `mlp_3x200_relu` learns strong training fits but also drifts substantially.
- BN + L2 is unstable and has a very large final-vs-best validation gap.

This is one of the clearest improvement opportunities. The current results are
likely underestimating what the best MLP variants can do, because the final
epoch model is used instead of the best validation checkpoint.

## Why LR Is Hard to Beat Here

Linear regression is a strong benchmark for this particular setup.

Reasons:

1. Financial daily returns are noisy and low signal-to-noise.
2. The target is an average future return, which smooths noise but also makes
   the relationship close to a low-amplitude linear problem.
3. Dense MLPs flatten the sequence, so they do not have an explicit temporal
   inductive bias. They see lagged returns as a wide feature vector.
4. The train/test split is chronological. Non-stationarity matters: patterns
   learned from older periods may not survive in the final test period.
5. LR is convex and stable. MLPs are non-convex, seed-sensitive, and easier to
   overfit.
6. The MLPs predict all 23 assets jointly. That can help if cross-asset effects
   are stable, but it can also increase variance if relationships shift.

The important point is that LR is not a weak baseline. Any neural model that
beats it is finding something meaningful.

## When the MLPs Beat LR

The MLPs beat LR mainly when `input_window=90`.

Best model by input window:

| input_window | best_mlp_mean | lr_mean  | delta     | pct       | mlp_wins |
| ------------ | ------------- | -------- | --------- | --------- | -------- |
| 5            | 0.005579      | 0.005405 | 0.000174  | 4.946436  | 0        |
| 10           | 0.005681      | 0.005473 | 0.000208  | 5.411765  | 0        |
| 30           | 0.005596      | 0.005647 | -0.000051 | -0.106234 | 2        |
| 90           | 0.005612      | 0.006147 | -0.000536 | -8.464667 | 4        |

Best model by output horizon:

| output_window | best_mlp_mean | lr_mean  | delta     | pct       | mlp_wins |
| ------------- | ------------- | -------- | --------- | --------- | -------- |
| 1             | 0.012827      | 0.012989 | -0.000163 | -1.005142 | 2        |
| 5             | 0.005796      | 0.005887 | -0.000090 | -1.317141 | 2        |
| 30            | 0.002448      | 0.002441 | 0.000008  | 0.543083  | 1        |
| 90            | 0.001396      | 0.001356 | 0.000041  | 3.566499  | 1        |

The signal is stronger by input length than by output horizon.

This suggests the main MLP advantage is not simply "predicting smoother long
horizons". It is more likely that long historical context gives the MLP enough
information to learn nonlinear cross-lag or cross-asset effects.

## Architecture Lessons

### Lesson 1: Regularization matters more than raw depth

`mlp_2x100_dropout` and `mlp_2x100_relu` have the same parameter count.

Dropout:

- Mean test MAE: `0.005618`
- Wins vs LR: `6 / 16`
- Best MLP cells: `15 / 16`

Plain 2x100:

- Mean test MAE: `0.006678`
- Wins vs LR: `1 / 16`
- Best MLP cells: `0 / 16`

The difference is regularization and fewer epochs, not architecture size.

### Lesson 2: More width/depth can help, but only partially

`mlp_3x200_relu` beats LR in 3 cells and is often second-best among MLPs. That
means capacity can help. But it is still worse than dropout in 15 of 16 cells.

The 3x200 model should be seen as evidence that richer nonlinear functions can
help, not as the best final architecture.

### Lesson 3: BatchNorm is risky in this time-series setup

The BN + L2 model has enormous errors in several cells. It is not just slightly
worse; it is structurally unreliable.

Possible causes:

- BatchNorm moving averages may not match the final test period.
- ReLU before BatchNorm may be a poor ordering.
- L2 + BN + Adam may require different learning-rate tuning.
- The model may need checkpointing because the best validation state is not the
  final state.

### Lesson 4: Flattened MLPs ignore sequence structure

Flattening makes a 90-day input a vector of 2,070 features. The model can learn
interactions, but it does not know that features are ordered in time unless it
learns that from data.

This is probably why the MLP needs longer windows to beat LR. A recurrent,
convolutional, attention, or temporal pooling model may use the same history
more efficiently.

## Practical Conclusions

### Is this kind of model capable of beating LR?

Yes, but not automatically.

The dropout MLP proves that dense neural models can beat LR on meaningful parts
of the grid:

- `90 -> 1`: `8.99%` better than LR
- `90 -> 5`: `8.55%` better than LR
- `90 -> 30`: `7.67%` better than LR
- `90 -> 90`: `8.65%` better than LR

But the same family loses all short-input-window cells:

- `5 -> *`: LR wins all 4 horizons.
- `10 -> *`: LR wins all 4 horizons.

So the correct conclusion is:

Dense MLPs can beat LR when given enough historical context and sufficient
regularization, but they are not yet a generally superior model family for this
task. LR remains the safer default baseline unless the use case specifically
uses long input windows, especially 90-day inputs.

### Which model should be carried forward?

Carry forward `mlp_2x100_dropout`.

It is the only MLP that is consistently competitive, and it is the best neural
candidate for downstream portfolio experiments.

Also keep `mlp_3x200_relu` as a secondary reference, because it occasionally
finds useful signal that dropout does not, especially at `30 -> 90`.

Do not carry forward `mlp_2x100_bn_l2` without redesign.

## Recommended Improvements

### 1. Add early stopping and restore best weights

This is the highest-priority fix.

Current issue:

- The notebooks report the minimum validation MAE.
- But the test prediction uses the final epoch weights.
- Many models reach best validation loss early and then get worse.

Recommended Keras callback:

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        min_delta=1e-6,
    )
]
```

Then train with:

```python
hist = model.fit(
    X_train,
    d.y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    shuffle=False,
    callbacks=callbacks,
)
```

Expected effect:

- Better test MAE for unregularized MLPs.
- Better stability for `mlp_3x200_relu`.
- Possibly much better behavior for BN + L2, if the final epoch is the problem.

### 2. Run multiple seeds

Current results use one seed: `RANDOM_SEED = 42`.

Neural models can vary meaningfully by initialization. A single run is not
enough to decide whether a 1% or 2% improvement is real.

Recommended:

- Run seeds: `7`, `21`, `42`, `123`, `2026`
- Report mean MAE, standard deviation, and confidence intervals per cell.
- Treat improvements below roughly 1% as inconclusive unless stable across
  seeds.

### 3. Use walk-forward validation

Current split:

- Train: older history
- Validation: last 10% of train
- Test: final 10%

This is acceptable for an initial benchmark, but financial data is
non-stationary. A single validation/test split can over-represent one market
regime.

Recommended:

- Use expanding-window validation.
- Example:
  - Train 1962-2005, validate 2006-2010
  - Train 1962-2010, validate 2011-2015
  - Train 1962-2015, validate 2016-2020
  - Train 1962-2020, test 2021-2026

This will show whether the MLP advantage is regime-specific.

### 4. Add a residual LR + MLP model

The cleanest way to exploit the benchmark is not to compete with it directly,
but to learn what it misses.

Recommended design:

```text
prediction = LR_prediction + MLP_residual_prediction
```

or a wide-and-deep network:

```text
linear branch: flattened input -> Dense(23, linear)
deep branch: flattened input -> Dense/ReLU/Dropout blocks -> Dense(23)
output: linear branch + deep branch
```

Why this is promising:

- LR is already strong.
- The MLP only needs to learn nonlinear residual structure.
- This reduces the chance that the MLP destroys the linear signal.
- It directly targets the question: "Can neural nets improve on LR?"

### 5. Tune dropout model, not every model equally

The dropout model is the best current candidate. Focus the search there.

Suggested search:

| Hyperparameter | Values |
| -------------- | ------ |
| hidden layers | 1, 2, 3 |
| units | 64, 100, 128, 200 |
| dropout | 0.05, 0.10, 0.20, 0.30 |
| learning rate | 1e-4, 3e-4, 1e-3 |
| batch size | 64, 128, 256 |
| epochs | high cap with early stopping |

The current result suggests `dropout=0.2` is useful, but it might be too strong
for short windows and too weak for long windows.

### 6. Rework BatchNorm experiment

Do not conclude that all normalization is bad. Conclude that this specific
BatchNorm setup is bad.

Better variants:

1. L2 without BatchNorm.
2. BatchNorm before activation:

```text
Dense(100, use_bias=False)
BatchNormalization()
Activation("relu")
```

3. LayerNormalization instead of BatchNormalization.
4. Lower learning rate.
5. Early stopping with restored best weights.

### 7. Add temporal architectures

Dense MLPs flatten time. For forecasting, that is a limitation.

Next model families to compare:

- Conv1D: local lag patterns, fewer parameters than flattening.
- GRU/LSTM: recurrent temporal state.
- Temporal CNN / TCN: stable long-context sequence modeling.
- Conv1D + GRU: local feature extraction plus temporal aggregation.
- Attention or transformer-lite model: cross-lag and cross-asset interactions.

Given the MLP result, the strongest hypothesis is:

Long input windows contain useful signal, but dense flattening is an inefficient
way to use it.

### 8. Improve target and preprocessing

Potential changes:

- Standardize returns per asset using train-only statistics.
- Winsorize extreme daily returns.
- Add volatility features.
- Add rolling mean/volatility/momentum features.
- Predict risk-adjusted return instead of raw average return.
- Predict direction or rank instead of raw return magnitude.
- Use asset-specific normalization so high-volatility assets do not dominate
  MAE.

Important: any transformation must be fit on train only to avoid leakage.

### 9. Evaluate economic value, not only MAE

MAE is useful, but portfolio decisions may care more about ranking assets than
absolute return error.

Add:

- Spearman rank correlation between predicted and realized asset returns.
- Top-k asset hit rate.
- Long-short spread return.
- Portfolio backtest for `output_window=90`.
- Turnover and transaction costs.
- Maximum drawdown and Sharpe ratio.

It is possible for a model to have similar MAE to LR but better ranking quality,
which could matter more for portfolio construction.

## Final Verdict

The current MLP experiments are valuable because they show a clear pattern:

1. Plain dense MLPs mostly do not beat linear regression.
2. Larger dense MLPs can find some useful nonlinear signal, but they overfit.
3. Dropout is the major improvement and makes the MLP competitive.
4. BatchNorm + L2 is not working in the current setup.
5. The MLP advantage appears mainly with long input windows, especially
   `input_window=90`.

Answer to the main question:

This kind of model can get better results than LR, but only with the right
regularization and enough historical context. The current best candidate,
`mlp_2x100_dropout`, is strong enough to justify further development, but not
strong enough to declare dense MLPs generally superior to LR.

The next best step is not to add more plain MLPs. The next best step is to
improve the experimental protocol and turn the dropout MLP into either:

1. an early-stopped, tuned dropout MLP, or
2. a residual model that learns corrections on top of LR.

That is the most direct path toward proving whether neural models can add value
over the linear benchmark.

