# dtf: NCCL layer for fault-tolerant training (guidance for Claude Code)

## Status and goal
This repo is an early scaffold: about 250 lines of C++ wrapping NCCL, with pybind11
bindings. Every Python module, test, example and doc is an empty placeholder (see
PROJECT_STRUCTURE.md).

The direction is failure handling for NCCL-based training. We WRAP PyTorch + NCCL;
we do NOT re-implement DDP/FSDP, tensor or pipeline parallelism, gradient
compression or custom all-reduce kernels.

The scope relative to the separate `goodput-fault-tolerant-training-orchestrator`
repo is not decided yet. Until `docs/design.md` and `docs/design-decisions.md` exist,
do not write feature code in `dtf/` or `csrc/`. Docs, hygiene and build fixes are fine.

## Layout
- csrc/       thin C++ NCCL layer (the only real code today)
- dtf/        Python package (empty placeholders)
- tests/      empty placeholders; real tests arrive with the first build PR
- bench/      (to be created) scripts that generate bench/RESULTS.md

## Commands (target; most don't work until the build/CI PR lands)
- pip install -e ".[dev]"   # builds dtf_C via CMake (needs CUDA + NCCL)
- pytest -m "not gpu"       # CPU tests; must pass before every commit once they exist
- pytest -m gpu             # only on a GPU host, only when asked
- ruff check . && mypy dtf  # lint and types (to be added to dev deps)

## Rules
- Never write benchmark numbers, scaling claims or feature claims by hand.
  README tables come only from bench/ scripts and name the GPU, driver, NCCL and
  PyTorch versions. If something isn't implemented, the README says "planned".
- Every NCCL error path must abort ALL communicators the process holds.
- Never call ncclCommDestroy on a communicator that saw an error; abort it.
- Thresholds (hang deadline, straggler factor) are relative to measured step
  time, configurable, and tested.
- One milestone slice per branch/PR; tests first; update docs and CHANGELOG.
- Ask before running anything that changes GPU or system state
  (nvidia-smi -lgc/-rgc, kill of processes you didn't start, sudo).
- The master booklet PDF is local reference material; never commit it.
