#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

mkdir -p model/cnn/executed
mkdir -p data/cnn/window_results data/cnn/history data/cnn/plots

for input_window in 5 10 30 90; do
  for output_window in 1 5 30 90; do
    result_file="data/cnn/window_results/cnn_input${input_window}_output${output_window}_results.csv"
    notebook="model/cnn/cnn_input${input_window}_output${output_window}.ipynb"

    if [ -f "$result_file" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
      echo "SKIP existing $result_file"
      continue
    fi

    echo "RUN $notebook"
    jupyter nbconvert --to notebook --execute "$notebook" \
      --output "cnn_input${input_window}_output${output_window}_executed.ipynb" \
      --output-dir model/cnn/executed \
      --ExecutePreprocessor.timeout=-1
  done
done

python - <<'PY'
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "model" / "cnn"))

from cnn_utils import aggregate_cnn_grid_results

results, comparison, matrix = aggregate_cnn_grid_results()

print("")
print("Número de modelos CNN:", len(results))
print(results[["input_window", "output_window", "MAE_train", "MAE_val", "MAE_test", "params", "epochs_trained"]].to_string(index=False))

print("")
print("Comparación vs LR:")
print(comparison[["input_window", "output_window", "MAE_test", "LR_MAE_test", "pct_delta_vs_lr"]].to_string(index=False))
PY
