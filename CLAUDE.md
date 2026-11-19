# dtf: NCCL layer for fault-tolerant training (guidance for Claude Code)

## Status and goal
A fault-aware NCCL communicator (non-blocking init with deadlines, async error polling,
abort-all registry, safe teardown, split), plus failure experiments whose written
findings feed the goodput orchestrator
(`goodput-fault-tolerant-training-orchestrator`). Today only a minimal C++ wrapper
exists. Read docs/design.md and docs/design-decisions.md before changing anything in
dtf/ or csrc/.

We WRAP PyTorch + NCCL; we do NOT re-implement DDP/FSDP, tensor or pipeline
parallelism, gradient compression or custom all-reduce kernels. Watchdog, fault
injection, classification, checkpointing and restart belong to goodput, not here
(ADR-3).

Build work (M2, M3) starts when goodput starts its M2 (ADR-8). Don't start it
unprompted.

## Layout
- csrc/            thin C++ NCCL layer (the only code today)
- dtf/             Python package; hosts the compiled dtf_C extension
- docs/            design.md, design-decisions.md; findings/ (one file per experiment X1–X8)
- tests/           (M2) CPU tests; GPU tests marked `gpu`
- bench/           (M3) scripts that generate bench/RESULTS.md

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
