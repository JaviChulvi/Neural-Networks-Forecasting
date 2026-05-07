# CNN Conv1D vs Linear Regression

## Scope

This section contains CNN Conv1D experiments for the 30 input / 5 output window.

## CNN Deep Conv1D

| model                       |   input_window |   output_window |   MAE_train |      MAE_val |   MAE_test |   params |   epochs_trained |   improvement_abs_vs_lr |   improvement_pct_vs_lr |
|:----------------------------|---------------:|----------------:|------------:|-------------:|-----------:|---------:|-----------------:|------------------------:|------------------------:|
| Linear_Regression_Benchmark |             30 |               5 |  0.00533732 | nan          | 0.00587674 |      nan |              nan |             0           |                 0       |
| CNN_Deep_Conv1D             |             30 |               5 |  0.00547655 |   0.00415256 | 0.0055767  |    93399 |               24 |             0.000300044 |                 5.10562 |

## CNN Hyperparameter Search

| model                       |   input_window |   output_window |   MAE_train |      MAE_val |   MAE_test |   params |   selected_trial_id | selection_metric   |   improvement_abs_vs_lr |   improvement_pct_vs_lr |
|:----------------------------|---------------:|----------------:|------------:|-------------:|-----------:|---------:|--------------------:|:-------------------|------------------------:|------------------------:|
| Linear_Regression_Benchmark |             30 |               5 |  0.00533732 | nan          | 0.00587674 |      nan |                 nan | nan                |             0           |                 0       |
| CNN_Optimized_RandomSearch  |             30 |               5 |  0.00547637 |   0.00415061 | 0.00557802 |    78327 |                   1 | MAE_val            |             0.000298728 |                 5.08322 |

The random search selected the model with the best validation MAE. The optimized model achieved a test MAE very close to the deeper CNN while using fewer parameters.
