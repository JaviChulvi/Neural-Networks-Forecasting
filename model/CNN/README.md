# CNN models

This folder contains the CNN work for the forecasting practice.

## Notebooks

1. `01_CNN_Deep_Conv1D.ipynb`  
   Trains and evaluates the selected deep CNN architecture for one input/output window combination. By default it uses `input_window=30` and `output_window=5`, matching the exploratory experiment already validated.

2. `02_CNN_Hyperparameter_Search.ipynb`  
   Runs a lightweight random search over CNN hyperparameters. The model is selected by validation MAE, and the test set is used only once for the final evaluation of the selected configuration.

## Outputs

Generated files are written under:

```text
model/CNN/outputs/
```

The output folders are intentionally ignored from the clean ZIP because trained models and plots can become large.
