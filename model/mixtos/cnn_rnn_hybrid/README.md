# Hybrid CNN-RNN models

This folder contains the hybrid models for the assigned input/output windows:

- input 10 / output 30
- input 10 / output 90
- input 30 / output 1
- input 30 / output 5

The implemented architectures combine convolutional and recurrent layers:

- `CNN_LSTM`
- `CNN_GRU`
- `CNN_BiGRU`

The main notebook logs all runs in MLflow and exports CSV summaries under `data/hybrid/`.

## Files

- `01_hybrid_cnn_rnn_grid.ipynb`: trains the hybrid models for the assigned windows.
- `02_hybrid_results_summary.ipynb`: summarizes the resulting CSV files and generates report-friendly tables.

## Outputs

Generated outputs are written to:

- `data/hybrid/hybrid_all_results.csv`
- `data/hybrid/hybrid_best_by_window.csv`
- `data/hybrid/hybrid_comparison_vs_lr.csv`
- `data/hybrid/hybrid_test_mae_matrix.csv`
- `data/hybrid/history/`
- `data/hybrid/plots/`

MLflow experiment name:

`hybrid_cnn_rnn_models`
