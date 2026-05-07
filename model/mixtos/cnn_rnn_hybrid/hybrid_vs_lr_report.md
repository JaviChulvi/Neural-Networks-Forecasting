# Hybrid CNN-RNN Models vs Linear Regression

## Scope

Hybrid CNN-RNN models were trained for the assigned windows: 10-30, 10-90, 30-1 and 30-5.

Architectures tested:

- CNN_GRU
- CNN_LSTM
- CNN_BiGRU

## Best model by test MAE

|   input_window |   output_window | model    |   MAE_train |     MAE_val |   MAE_test |   LR_MAE_test |   pct_delta_vs_lr |   params |   epochs_trained |
|---------------:|----------------:|:---------|------------:|------------:|-----------:|--------------:|------------------:|---------:|-----------------:|
|             10 |              30 | CNN_GRU  |  0.00220192 | 0.00169853  | 0.00231971 |    0.00235841 |         -1.64088  |    35351 |               35 |
|             10 |              90 | CNN_LSTM |  0.0012673  | 0.000926007 | 0.00126981 |    0.00128239 |         -0.980589 |    43415 |               18 |
|             30 |               1 | CNN_LSTM |  0.0118412  | 0.00903741  | 0.012244   |    0.0129242  |         -5.263    |    43415 |               15 |
|             30 |               5 | CNN_GRU  |  0.00547891 | 0.00415408  | 0.00558128 |    0.00587674 |         -5.02772  |    35351 |               17 |

Negative values in `pct_delta_vs_lr` indicate an improvement over the linear regression benchmark.

## Number of trained models

Total trained hybrid models: **12**.
