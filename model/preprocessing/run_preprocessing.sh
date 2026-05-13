#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
mkdir -p model/preprocessing/executed

jupyter nbconvert --to notebook --execute model/preprocessing/01_yahoo_ohlcv_audit.ipynb   --output 01_yahoo_ohlcv_audit_executed.ipynb   --output-dir model/preprocessing/executed   --ExecutePreprocessor.timeout=-1

jupyter nbconvert --to notebook --execute model/preprocessing/02_activity_bars_yahoo.ipynb   --output 02_activity_bars_yahoo_executed.ipynb   --output-dir model/preprocessing/executed   --ExecutePreprocessor.timeout=-1
