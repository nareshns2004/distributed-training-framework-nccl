# Contributing

Thank you for your interest in contributing to `distributed-training-framework-nccl`.

## Getting Started

1. Fork the repository and clone your fork.
2. Set up the development environment:

```bash
pip install -e ".[dev]"
pre-commit install
```

3. Create a branch for your change:

```bash
git checkout -b feat/your-feature-name
```

## Development Setup

### Build from source (C++/CUDA)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DNCCL_ROOT=/path/to/nccl
make -j$(nproc)
cd ..
```

### Running tests

```bash
# Unit tests (no GPU required)
pytest tests/unit/

# Integration tests (requires 2+ GPUs)
bash scripts/run_tests.sh
```

### Code formatting

```bash
# Python
black dtf/ tests/
ruff dtf/ tests/

# C++/CUDA
bash scripts/format.sh   # runs clang-format
```

Or run all formatters at once via pre-commit:

```bash
pre-commit run --all-files
```

## Contribution Guidelines

### What we welcome

- Bug fixes with a clear reproduction case
- Performance improvements with benchmark data
- New collective operations or compression algorithms
- Documentation improvements
- Test coverage additions

### What to discuss first

For larger changes — new parallelism strategies, breaking API changes, significant architecture shifts — please open an issue first to align on direction before writing code.

### Pull request process

1. Ensure all tests pass and new code has tests.
2. Update documentation if you change public APIs.
3. Add an entry to `CHANGELOG.md` under `[Unreleased]`.
4. Keep commits focused and write clear commit messages.
5. Open a PR with a concise description of the change and why it's needed.

### Commit message format

```
<type>: <short summary>

<optional body>
```

Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `build`, `ci`.

Example: `perf: fuse cast and quantize kernels in compression.cu`

## Reporting Bugs

Please include: OS, CUDA version, NCCL version, Python version, a minimal reproducer, and the full stack trace.

## Security

Do not open public issues for security vulnerabilities. Email [security@your-org.com](mailto:security@your-org.com) instead.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
