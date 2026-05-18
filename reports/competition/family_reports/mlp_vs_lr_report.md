# MLP vs Linear Regression Report

Generated from the current fixed MLP CSV files in `data/mlp/`.

## Main Conclusion

The best current fixed MLP by mean test MAE is `mlp_2x100_dropout`.

- Best fixed MLP mean test MAE: `0.005416`
- LR mean test MAE: `0.005668`
- Mean delta vs LR: `-0.000252` (-4.44%)
- Per-window best MLP beats LR: `16 / 16`

## Model Ranking

| model_name | mean_test | median_test | best_test | worst_test | mean_delta_vs_lr | wins_vs_lr | best_mlp_cells | avg_epochs | avg_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_2x100_dropout | 0.005416 | 0.004003 | 0.001371 | 0.012293 | -0.000252 | 11 | 7 | 200.000000 | 91701.500000 |
| mlp_3x200_relu | 0.005419 | 0.004006 | 0.001377 | 0.012299 | -0.000249 | 11 | 2 | 200.000000 | 242026.500000 |
| mlp_4x100_gelu_dropout_l2 | 0.005533 | 0.004024 | 0.001268 | 0.013049 | -0.000136 | 14 | 7 | 339.750000 | 111901.500000 |
| mlp_3x128_gelu_dropout_l2 | 0.005614 | 0.004033 | 0.001345 | 0.013443 | -0.000054 | 8 | 0 | 500.000000 | 137032.500000 |
| mlp_2x100_relu | 0.006047 | 0.004639 | 0.001705 | 0.013396 | 0.000378 | 5 | 0 | 500.000000 | 91701.500000 |
| mlp_1x64_relu | 0.006979 | 0.004884 | 0.001433 | 0.017933 | 0.001311 | 1 | 0 | 500.000000 | 52792.500000 |
| mlp_1x100_relu | 0.007011 | 0.004908 | 0.001455 | 0.018371 | 0.001343 | 0 | 0 | 500.000000 | 81601.500000 |
| mlp_2x100_bn_l2 | 0.007970 | 0.006551 | 0.002116 | 0.017400 | 0.002302 | 2 | 0 | 200.000000 | 92501.500000 |

## Best MLP Per Window

| input_window | output_window | model_name | MAE_test | MAE_test_lr | delta_vs_lr | pct_delta_vs_lr |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 1 | mlp_3x200_relu | 0.012273 | 0.012384 | -0.000111 | -0.896220 |
| 5 | 5 | mlp_4x100_gelu_dropout_l2 | 0.005598 | 0.005625 | -0.000026 | -0.466603 |
| 5 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002320 | 0.002340 | -0.000020 | -0.864923 |
| 5 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001268 | 0.001271 | -0.000004 | -0.281813 |
| 10 | 1 | mlp_2x100_dropout | 0.012270 | 0.012554 | -0.000284 | -2.265491 |
| 10 | 5 | mlp_2x100_dropout | 0.005609 | 0.005698 | -0.000089 | -1.554030 |
| 10 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002319 | 0.002358 | -0.000039 | -1.650715 |
| 10 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001268 | 0.001282 | -0.000015 | -1.136798 |
| 30 | 1 | mlp_3x200_relu | 0.012282 | 0.012924 | -0.000642 | -4.968064 |
| 30 | 5 | mlp_2x100_dropout | 0.005611 | 0.005877 | -0.000265 | -4.515383 |
| 30 | 30 | mlp_4x100_gelu_dropout_l2 | 0.002360 | 0.002436 | -0.000076 | -3.133119 |
| 30 | 90 | mlp_4x100_gelu_dropout_l2 | 0.001285 | 0.001351 | -0.000066 | -4.895765 |
| 90 | 1 | mlp_2x100_dropout | 0.012293 | 0.014095 | -0.001802 | -12.787344 |
| 90 | 5 | mlp_2x100_dropout | 0.005623 | 0.006348 | -0.000725 | -11.418789 |
| 90 | 30 | mlp_2x100_dropout | 0.002397 | 0.002628 | -0.000231 | -8.795829 |
| 90 | 90 | mlp_2x100_dropout | 0.001389 | 0.001518 | -0.000129 | -8.471753 |

## Best MLP MAE Matrix

| output_window | 5 | 10 | 30 | 90 |
| --- | --- | --- | --- | --- |
| 1 | 0.012273 | 0.012270 | 0.012282 | 0.012293 |
| 5 | 0.005598 | 0.005609 | 0.005611 | 0.005623 |
| 30 | 0.002320 | 0.002319 | 0.002360 | 0.002397 |
| 90 | 0.001268 | 0.001268 | 0.001285 | 0.001389 |

## Winning MLP Architecture Matrix

| output_window | 5 | 10 | 30 | 90 |
| --- | --- | --- | --- | --- |
| 1 | mlp_3x200_relu | mlp_2x100_dropout | mlp_3x200_relu | mlp_2x100_dropout |
| 5 | mlp_4x100_gelu_dropout_l2 | mlp_2x100_dropout | mlp_2x100_dropout | mlp_2x100_dropout |
| 30 | mlp_4x100_gelu_dropout_l2 | mlp_4x100_gelu_dropout_l2 | mlp_4x100_gelu_dropout_l2 | mlp_2x100_dropout |
| 90 | mlp_4x100_gelu_dropout_l2 | mlp_4x100_gelu_dropout_l2 | mlp_4x100_gelu_dropout_l2 | mlp_2x100_dropout |

## Parameter Counts

| model_name | 5 | 10 | 30 | 90 |
| --- | --- | --- | --- | --- |
| mlp_1x100_relu | 14154 | 25884 | 72804 | 213564 |
| mlp_1x64_relu | 9150 | 16740 | 47100 | 138180 |
| mlp_2x100_bn_l2 | 25054 | 36784 | 83704 | 224464 |
| mlp_2x100_dropout | 24254 | 35984 | 82904 | 223664 |
| mlp_2x100_relu | 24254 | 35984 | 82904 | 223664 |
| mlp_3x128_gelu_dropout_l2 | 51070 | 66020 | 125820 | 305220 |
| mlp_3x200_relu | 108454 | 131684 | 224604 | 503364 |
| mlp_4x100_gelu_dropout_l2 | 44454 | 56184 | 103104 | 243864 |

## Interpretation

- `mlp_2x100_dropout` is the strongest fixed MLP by mean test MAE.
- The per-window winner is not always the same architecture, so the global ranking and
  the best-per-window table should be read together.
- The newest `mlp_4x100_gelu_dropout_l2` rows include the internal Keras `Normalization`
  layer parameters and the usable-checkpoint metadata, so its parameter count is slightly
  higher than the older report version.
- Use `model/mlp/99_compare_mlp_vs_lr.ipynb` for exploratory notebook views, and this
  generator for the Markdown report consumed by `reports/competition/`.
