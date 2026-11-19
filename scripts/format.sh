#!/bin/bash
set -e

echo "Formatting Python files..."
black dtf

echo "Formatting C++/CUDA files..."
find csrc/ -iname '*.h' -o -iname '*.cpp' -o -iname '*.cu' -o -iname '*.cuh' | xargs clang-format -i

echo "Formatting complete."
