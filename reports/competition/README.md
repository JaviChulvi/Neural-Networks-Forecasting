# Competition results

This folder groups the final competition outputs requested in the assignment:

- Global best model per input/output window.
- Individual family reports: MLP, CNN, RNN and mixed models.
- Matrices and CSV outputs used to compare MAE values.

The training notebooks and scripts remain in `model/`.

## Regeneration

Run these commands from the repository root to refresh the competition package:

```bash
python model/mlp/generate_mlp_vs_lr_report.py
python model/rnn/generate_rnn_vs_lr_report.py
python model/mixtos/generate_mixtos_report.py
python reports/competition/generate_best_model_per_window.py
```

The global generator reads the current MLP CSV files from `data/mlp/` and
updates `mlp_best_family_results_reference.csv` before building the global
best-model report.
