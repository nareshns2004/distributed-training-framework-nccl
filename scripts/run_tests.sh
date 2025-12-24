#!/bin/bash
set -e

echo "Running unit tests..."
pytest tests/unit

echo "Running integration tests..."
pytest tests/integration

echo "All tests passed."
