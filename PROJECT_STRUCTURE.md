# Project Structure

```
distributed-training-framework-nccl/
├── README.md                  # Status, scope and roadmap
├── CLAUDE.md                  # Rules for Claude Code (no hand-written claims)
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── .gitignore
│
├── CMakeLists.txt             # Top-level CMake build (known issues; fixed in M2)
├── setup.py                   # Python package build; calls CMake
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── docs/
│   ├── design.md              # Scope, architecture, failure experiments, measurements
│   └── design-decisions.md    # ADR-1 … ADR-8
│
├── csrc/                      # C++/CUDA source: the only code today
│   ├── CMakeLists.txt
│   ├── include/dtf/
│   │   ├── communicator.h     # NCCL communicator wrapper
│   │   ├── collective_ops.h   # all-reduce, broadcast, all-gather
│   │   ├── process_group.h    # Owns a Communicator
│   │   └── utils.h            # DTF_CHECK, NCCL_CHECK
│   ├── core/
│   │   ├── communicator.cpp
│   │   ├── collective_ops.cpp
│   │   └── process_group.cpp
│   ├── cuda/
│   │   ├── memory_utils.cu    # cudaMalloc / cudaFree helpers
│   │   └── all_reduce.cu      # comment only; defers to NCCL
│   └── bindings/
│       └── pybind.cpp         # dtf_C module: get_unique_id, Communicator, ProcessGroup
│
├── dtf/
│   └── __init__.py            # empty; the compiled dtf_C extension is installed here
│
├── docker/                    # Dockerfile (CUDA 11.8), Dockerfile.dev, docker-compose.yml
└── scripts/                   # build.sh, run_tests.sh, format.sh
```

Planned in M2–M3 (docs/design.md §7): `tests/` (CPU and GPU), `bench/` (scripts that
write `bench/RESULTS.md`), `docs/findings/` (one write-up per failure experiment) and
`.github/workflows/` (CI).
