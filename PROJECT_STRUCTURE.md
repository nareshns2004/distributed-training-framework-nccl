# Project Structure

Files marked **(empty)** are zero-byte placeholders left from an earlier, broader plan.
They are not features. See the README for what is implemented.

```
distributed-training-framework-nccl/
├── README.md                  # Honest status page
├── CLAUDE.md                  # Rules for Claude Code (no hand-written claims)
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── .gitignore
│
├── CMakeLists.txt             # Top-level CMake build (known issues; see README)
├── setup.py                   # Python package build; calls CMake
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── csrc/                      # C++/CUDA source: the only real code
│   ├── CMakeLists.txt
│   ├── include/dtf/
│   │   ├── communicator.h     # NCCL communicator wrapper
│   │   ├── collective_ops.h   # all-reduce, broadcast, all-gather
│   │   ├── process_group.h    # Owns a Communicator
│   │   ├── utils.h            # DTF_CHECK, NCCL_CHECK
│   │   └── compression.h      # (empty)
│   ├── core/
│   │   ├── communicator.cpp
│   │   ├── collective_ops.cpp
│   │   └── process_group.cpp
│   ├── cuda/
│   │   ├── memory_utils.cu    # cudaMalloc / cudaFree helpers
│   │   ├── all_reduce.cu      # comment only; defers to NCCL
│   │   ├── compression.cu     # (empty)
│   │   └── cast_fp16_bf16.cu  # (empty)
│   └── bindings/
│       └── pybind.cpp         # dtf_C module: get_unique_id, Communicator, ProcessGroup
│
├── dtf/                       # Python package: every module is (empty)
│   ├── __init__.py, distributed.py, process_group.py, collective_ops.py
│   ├── checkpoint/            # __init__.py, save.py, load.py
│   ├── compression/           # __init__.py, topk.py, quantize.py, powersgd.py
│   ├── parallelism/           # __init__.py, data_parallel.py, tensor_parallel.py, pipeline_parallel.py
│   └── utils/                 # __init__.py, env.py, logging.py, profiling.py
│
├── launcher/                  # __init__.py, launch.py: (empty)
├── tests/                     # all (empty)
│   ├── unit/                  # test_communicator, test_collective_ops, test_compression, test_checkpoint
│   ├── integration/           # test_data_parallel, test_tensor_parallel, test_multi_node
│   └── benchmarks/            # bench_allreduce, bench_throughput
├── examples/                  # all (empty)
├── docs/                      # api, architecture, benchmarks, configuration: all (empty)
│
├── docker/                    # Dockerfile (CUDA 11.8), Dockerfile.dev, docker-compose.yml
└── scripts/                   # build.sh, run_tests.sh, format.sh
```
