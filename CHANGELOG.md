# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Only work that exists in the repository goes under a release heading; everything
else lives under **Planned**.

## [Unreleased]

### Added
- `CLAUDE.md` with project rules: no hand-written benchmark numbers or feature claims,
  and NCCL error-path rules.
- `docs/design.md` and `docs/design-decisions.md` (ADR-1 to ADR-8). They scope the repo
  to a fault-aware NCCL layer plus failure experiments for the goodput orchestrator.
- A `gpu` pytest marker.

### Changed
- Rewrote the README as an honest status page: scope, what exists, known issues and
  roadmap.
- `PROJECT_STRUCTURE.md` lists only files that exist.
- `setup.py` packages only `dtf`; `scripts/run_tests.sh` and `scripts/format.sh` match
  the new layout.
- Fixed placeholder metadata (clone URL, author email, NOTICE, security contact).

### Removed
- 40 empty placeholder files from the earlier plan: `dtf/` submodules (parallelism,
  compression, checkpoint, utils), `launcher/`, `examples/`, `tests/`, the empty
  `docs/*.md`, `compression.h`, `compression.cu` and `cast_fp16_bf16.cu`.
- The README benchmark table (GPT-2 1.5B and LLaMA-7B throughput and scaling). No code
  in this repository produced those numbers.
- The README quick-start, which used APIs that don't exist (`dtf.init`,
  `dtf.DistributedModel`, `dtf-launch`).
- The tracked `dtf.egg-info/` build output.

### Fixed
- `gitignore` renamed to `.gitignore`, so its rules now apply.

## Existing code (initial scaffold, 25 Dec 2025)

- C++ `Communicator` wrapping `ncclGetUniqueId`, `ncclCommInitRank` and
  `ncclCommDestroy`.
- C++ `CollectiveOps` wrapping `ncclAllReduce`, `ncclBroadcast` and `ncclAllGather`
  (not yet exposed to Python).
- C++ `ProcessGroup` owning a `Communicator`.
- pybind11 module `dtf_C` exposing `get_unique_id`, `Communicator` and `ProcessGroup`.
- CMake + `setup.py` build (not yet verified on a CUDA host) and Dockerfiles.

## Planned

These were previously listed as "Added" but don't exist. They are kept here only as a
record. Several items have been dropped, as marked.

- Collectives bound to Python with a CUDA stream argument; reduce-scatter.
- Fault-aware NCCL layer: non-blocking init with a deadline, async error polling,
  `ncclCommAbort`, safe teardown.
- Checkpoint save/load with resume.
- Unit and integration test suite; CI.
- Launcher CLI.
- Docker images verified for CUDA 12.x.
- *Dropped:* custom all-reduce and FP16/BF16 cast kernels; gradient compression (Top-K,
  PowerSGD, quantisation); from-scratch data, tensor and pipeline parallelism; JAX
  support. PyTorch and NCCL already provide these.
