#!/bin/bash
set -e

# CPU tests by default; pass -m gpu on a GPU host for the GPU tests.
# There are no tests yet: they arrive with milestone M2 (docs/design.md §7).
pytest -m "not gpu" "$@"
