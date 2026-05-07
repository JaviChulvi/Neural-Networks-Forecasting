# CNN Conv1D Models

This folder contains the CNN Conv1D experiments developed for the forecasting workshop.

## Contents

- 01_CNN_Deep_Conv1D.ipynb  
  Trains and evaluates a deeper Conv1D model for the 30 input / 5 output window.

- 02_CNN_Hyperparameter_Search.ipynb  
  Performs a random-search hyperparameter experiment for the CNN Conv1D model.

- cnn_vs_lr_report.md  
  Summary report comparing CNN results against the linear regression benchmark.

- manifest.txt  
  Short inventory of the files included in this section.

## Outputs

Generated result files are stored in:

- data/cnn/
- data/cnn_search/

data/cnn/ contains the selected CNN Deep Conv1D results, comparison against the linear regression benchmark, training history and loss curve.

data/cnn_search/ contains the random-search hyperparameter results, best configuration and trial histories.

## Main result

For the 30 input / 5 output window, the CNN Deep Conv1D model achieved a lower test MAE than the linear regression benchmark.

The random-search optimized CNN achieved a very similar test MAE with fewer parameters.
