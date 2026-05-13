#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python model/preprocessing/03_build_preprocessed_sequences.py
python model/preprocessing/update_preprocessing_report_sequences.py
