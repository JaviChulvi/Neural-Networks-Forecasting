rnn# MLP vs Linear Regression Report

Generated from the current fixed MLP CSV files in `data/mlp/`.

## Main Conclusion

The best current fixed MLP is `mlp_4x100_gelu_dropout_l2`.

- Best MLP mean test MAE: `0.005371`
- LR mean test MAE: `0.005668`
- Mean delta vs LR: `-0.000297`
- Cells where the best fixed MLP beats LR: `16 / 16`

The newer GELU + dropout + L2 models now outperform the earlier ReLU baselines. The old best model, `mlp_2x100_dropout`, is no longer the best fixed MLP once `08` and `09` are included.

## Model Ranking

| model_name | mean_test | median_test | best_test | worst_test | mean_delta_vs_lr | wins_vs_lr | best_mlp_cells | avg_epochs | avg_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_4x100_gelu_dropout_l2 | 0.005371 | 0.003948 | 0.001263 | 0.012487 | -0.000297 | 16 | 11 | 500.000000 | 110348 |
| mlp_3x128_gelu_dropout_l2 | 0.005389 | 0.003988 | 0.001341 | 0.012249 | -0.000279 | 11 | 5 | 500.000000 | 135479 |
| mlp_2x100_dropout | 0.005618 | 0.004121 | 0.001387 | 0.012971 | -0.000050 | 6 | 0 | 200.000000 | 90148 |
| mlp_3x200_relu | 0.006165 | 0.004479 | 0.001373 | 0.014438 | 0.000496 | 3 | 0 | 200.000000 | 240473 |
| mlp_2x100_relu | 0.006678 | 0.004705 | 0.001425 | 0.015833 | 0.001009 | 1 | 0 | 500.000000 | 90148 |
| mlp_1x64_relu | 0.006862 | 0.004597 | 0.001483 | 0.016718 | 0.001194 | 1 | 0 | 500.000000 | 51239 |
| mlp_1x100_relu | 0.006970 | 0.004950 | 0.001611 | 0.017497 | 0.001302 | 0 | 0 | 500.000000 | 80048 |
| mlp_2x100_bn_l2 | 0.019015 | 0.008115 | 0.002770 | 0.140086 | 0.013346 | 1 | 0 | 200.000000 | 90948 |

## Best MLP Per Window

| input_window | output_window | model_name | MAE_test | MAE_test_lr | delta_vs_lr | pct_delta_vs_lr |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | mlp_3x128_gelu_dropout_l2 | 0.012224 | 0.012384 | -0.000159 | -1.285952 |
| 5 | 5 | mlp_4x100_gelu_dropout_l2 | 0.005574 | 0.005625 | -0.000051 | -0.910257 |
| 5 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002321 | 0.002340 | -0.000019 | -0.827671 |
| 5 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001266 | 0.001271 | -0.000005 | -0.386101 |
| 10 | 1 | mlp_3x128_gelu_dropout_l2 | 0.012225 | 0.012554 | -0.000329 | -2.618626 |
| 10 | 5 | mlp_4x100_gelu_dropout_l2 | 0.005573 | 0.005698 | -0.000124 | -2.184385 |
| 10 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002321 | 0.002358 | -0.000038 | -1.607422 |
| 10 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001263 | 0.001282 | -0.000020 | -1.533588 |
| 30 | 1 | mlp_3x128_gelu_dropout_l2 | 0.012232 | 0.012924 | -0.000692 | -5.357347 |
| 30 | 5 | mlp_4x100_gelu_dropout_l2 | 0.005574 | 0.005877 | -0.000303 | -5.159016 |
| 30 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002323 | 0.002436 | -0.000113 | -4.652721 |
| 30 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001264 | 0.001351 | -0.000087 | -6.450514 |
| 90 | 1 | mlp_3x128_gelu_dropout_l2 | 0.012249 | 0.014095 | -0.001847 | -13.100387 |
| 90 | 5 | mlp_3x128_gelu_dropout_l2 | 0.005605 | 0.006348 | -0.000744 | -11.712643 |
| 90 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002323 | 0.002628 | -0.000306 | -11.627320 |
| 90 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001268 | 0.001518 | -0.000250 | -16.465216 |

## Parameter Counts

| model_name | 5 | 10 | 30 | 90 |
| --- | --- | --- | --- | --- |
| mlp_1x100_relu | 13923 | 25423 | 71423 | 209423 |
| mlp_1x64_relu | 8919 | 16279 | 45719 | 134039 |
| mlp_2x100_bn_l2 | 24823 | 36323 | 82323 | 220323 |
| mlp_2x100_dropout | 24023 | 35523 | 81523 | 219523 |
| mlp_2x100_relu | 24023 | 35523 | 81523 | 219523 |
| mlp_3x128_gelu_dropout_l2 | 50839 | 65559 | 124439 | 301079 |
| mlp_3x200_relu | 108223 | 131223 | 223223 | 499223 |
| mlp_4x100_gelu_dropout_l2 | 44223 | 55723 | 101723 | 239723 |

## Interpretation

- `mlp_3x128_gelu_dropout_l2` is the best fixed model by mean test MAE.
- `mlp_4x100_gelu_dropout_l2` is very close and wins several individual windows, so it remains a useful secondary candidate.
- The main improvement comes from the combination of GELU activations, dropout, L2 regularization, and early stopping with restored best weights.
- The strongest improvements versus LR remain in longer input-window regimes, especially `input_window=90`.
- `mlp_2x100_bn_l2` remains unstable and should not be carried forward without redesign.

## Next Steps

- Rerun the top GELU models with multiple random seeds before choosing a final production candidate.
- Keep `mlp_2x100_dropout` as the old regularized ReLU reference.
- Use `99_compare_mlp_vs_lr.ipynb` as the source notebook for regenerating this report after future fixed-model CSVs are added.
