#!/usr/bin/env bash
cd "$(dirname "$0")"
TIDY=$(command -v clang-tidy-20 || command -v clang-tidy || true)
if [ -z "$TIDY" ]; then echo 'clang-tidy not found; install clang-tidy-20 or clang-tidy'; exit 1; fi
FAIL=0
for f in $(find src include benchmarks tests -type f \( -name '*.cpp' -o -name '*.hpp' \)); do
  if ! "$TIDY" "$f" -p build --warnings-as-errors='*' --header-filter='^(?!.*_deps).*' 2>&1; then FAIL=1; fi
done
exit $FAIL
