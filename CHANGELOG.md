# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding
- NCCL communicator wrapper (`csrc/core/communicator.cpp`)
- Python bindings via pybind11
- AllReduce, AllGather, Broadcast, ReduceScatter collective ops
- Custom CUDA kernels for gradient compression
- Mixed-precision (FP16/BF16) cast kernels
- Data parallelism (`dtf.parallelism.data_parallel`)
- Tensor parallelism (`dtf.parallelism.tensor_parallel`)
- Pipeline parallelism (`dtf.parallelism.pipeline_parallel`)
- Top-K gradient sparsification
- PowerSGD low-rank compression
- Checkpoint save/load with fault tolerance
- `dtf-launch` multi-node launcher CLI
- Docker images for CUDA 11.8 and 12.x
- Unit and integration test suite

## [0.1.0] - TBD

Initial public release.
