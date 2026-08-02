#!/usr/bin/env bash
set -euo pipefail

experiment_root=$(cd "$(dirname "$0")" && pwd)
python_bin="$experiment_root/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "Missing .venv; create it and install requirements.txt first." >&2
  exit 1
fi

"$python_bin" "$experiment_root/src/prepare_data.py"
"$python_bin" "$experiment_root/src/run_experiments.py" "$@"
"$python_bin" "$experiment_root/src/run_experiments.py" \
  --config "$experiment_root/config/stress_test.json" "$@"
"$python_bin" "$experiment_root/src/analyze_results.py"
"$python_bin" "$experiment_root/src/generate_report.py"
