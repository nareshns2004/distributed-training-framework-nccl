# Project Structure

```
distributed-training-framework-nccl/
│
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── NOTICE
├── .gitignore
├── .clang-format
├── .pre-commit-config.yaml
│
├── CMakeLists.txt                    # Top-level CMake build
├── setup.py                         # Python package build (calls CMake)
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── csrc/                            # C++/CUDA source
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── dtf/
│   │   │   ├── communicator.h       # NCCL communicator wrapper
│   │   │   ├── collective_ops.h     # AllReduce, Broadcast, etc.
│   │   │   ├── process_group.h      # Distributed process group
│   │   │   ├── compression.h        # Gradient compression interfaces
│   │   │   └── utils.h
│   │
│   ├── core/
│   │   ├── communicator.cpp
│   │   ├── collective_ops.cpp
│   │   └── process_group.cpp
│   │
│   ├── cuda/
│   │   ├── all_reduce.cu            # Custom AllReduce kernels
│   │   ├── compression.cu           # Gradient sparsification / quantization
│   │   ├── cast_fp16_bf16.cu        # Mixed-precision cast kernels
│   │   └── memory_utils.cu
│   │
│   └── bindings/
│       └── pybind.cpp               # pybind11 Python bindings
│
├── dtf/                             # Python package
│   ├── __init__.py
│   ├── distributed.py               # High-level DistributedModel wrapper
│   ├── process_group.py             # Python process group management
│   ├── collective_ops.py            # Python collective op wrappers
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── topk.py                  # Top-K gradient sparsification
│   │   ├── quantize.py              # 1-bit / 8-bit quantization
│   │   └── powersgd.py              # PowerSGD low-rank approximation
│   ├── parallelism/
│   │   ├── __init__.py
│   │   ├── data_parallel.py         # Data parallelism
│   │   ├── tensor_parallel.py       # Tensor parallelism
│   │   └── pipeline_parallel.py     # Pipeline parallelism
│   ├── checkpoint/
│   │   ├── __init__.py
│   │   ├── save.py
│   │   └── load.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── profiling.py
│       └── env.py
│
├── launcher/                        # dtf-launch CLI tool
│   ├── __init__.py
│   └── launch.py
│
├── tests/
│   ├── unit/
│   │   ├── test_communicator.py
│   │   ├── test_collective_ops.py
│   │   ├── test_compression.py
│   │   └── test_checkpoint.py
│   ├── integration/
│   │   ├── test_data_parallel.py
│   │   ├── test_tensor_parallel.py
│   │   └── test_multi_node.py
│   └── benchmarks/
│       ├── bench_allreduce.py
│       └── bench_throughput.py
│
├── examples/
│   ├── gpt_data_parallel.py         # GPT training with data parallelism
│   ├── llama_tensor_parallel.py     # LLaMA with tensor parallelism
│   ├── pipeline_parallel_basic.py
│   └── gradient_compression.py
│
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   └── docker-compose.yml
│
├── scripts/
│   ├── build.sh
│   ├── run_tests.sh
│   └── format.sh
│
└── docs/
    ├── architecture.md
    ├── api.md
    ├── configuration.md
    └── benchmarks.md
```
