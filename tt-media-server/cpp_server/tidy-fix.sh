#!/usr/bin/env bash
# Auto-fix readability-identifier-naming and other clang-tidy checks.
# Run ./build.sh first to ensure compile_commands.json exists.
#
# Usage:
#   ./tidy-fix.sh                    # fix all files
#   ./tidy-fix.sh path/to/file.hpp   # fix single file
#
# For header+impl pairs: headers are fixed first; .cpp uses --fix-errors
# so fixes apply even when the header was already changed.
set -e
cd "$(dirname "$0")"
TIDY=$(command -v clang-tidy-20 || command -v clang-tidy || true)
if [ -z "$TIDY" ]; then
  echo 'clang-tidy not found; install clang-tidy-20 or clang-tidy'
  exit 1
fi
if [ ! -d build ]; then
  echo 'build/ missing; run ./build.sh first'
  exit 1
fi
HF='^(?!.*_deps).*'

fix_hpp() {
  "$TIDY" "$1" -p build --fix --header-filter="$HF" 2>/dev/null || true
}
fix_cpp() {
  "$TIDY" "$1" -p build --fix --fix-errors --header-filter="$HF" 2>/dev/null || true
}

if [ $# -gt 0 ]; then
  for f in "$@"; do
    [[ "$f" == *.hpp ]] && fix_hpp "$f" || fix_cpp "$f"
  done
else
  echo "Fixing headers..."
  for f in $(find src include benchmarks tests -type f -name '*.hpp'); do
    fix_hpp "$f"
  done
  echo "Fixing sources..."
  for f in $(find src benchmarks tests -type f -name '*.cpp'); do
    fix_cpp "$f"
  done
fi
echo "Done. Run ./tidy.sh to verify."
