#!/usr/bin/env bash
# Run all Docker-side (bpy-free) unit tests. Usage: bash run_unit_tests.sh
set -e
cd "$(dirname "$0")"
fail=0
for t in tests/unit/test_*.py; do
  echo "=== $t ==="
  python3 "$t" || fail=1
done
exit $fail
