#!/bin/bash
# Dupla kattintással lefuttatja a projekt összes tesztjét; az eredmény: tests_output.log
cd "$(dirname "$0")" || exit 1
{
  echo "=== $(date) ==="
  .venv/bin/python --version
  .venv/bin/python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
  .venv/bin/python -m pytest -q tests 2>&1
  echo "EXIT CODE: $?"
} > tests_output.log 2>&1
echo "Kész – az eredmény a tests_output.log fájlban. Ez az ablak bezárható."
