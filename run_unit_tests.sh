#!/usr/bin/env bash
# Run all Docker-side (bpy-free) unit tests. Usage: bash run_unit_tests.sh
set -eu
cd "$(dirname "$0")"

if [ -x .venv/bin/python ] && [ -z "${PYTHON:-}" ]; then
  PYTHON=.venv/bin/python
else
  PYTHON=${PYTHON:-python}
fi

"$PYTHON" -c 'import cv2, numpy, PIL, shapely' || {
  printf '%s\n' "Missing test dependencies. Install requirements-dev.txt first." >&2
  exit 2
}

fail=0
for t in tests/unit/test_*.py; do
  printf '=== %s ===\n' "$t"
  "$PYTHON" "$t" || fail=1
done
exit $fail
