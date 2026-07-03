#!/bin/bash
set -e

echo "Building distributed-training-framework-nccl (dtf)..."

# Build the C++ extension and Python package in-place
pip install -e . -v

echo "Build complete."
