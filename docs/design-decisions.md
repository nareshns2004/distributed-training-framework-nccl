# Design decisions

Short ADRs. Each records the decision, the alternative rejected, and the cost accepted.
All were accepted on 19 Nov 2026.

## ADR-1: Wrap PyTorch + NCCL; don't re-implement parallelism
- **Decision:** dtf does not implement DDP/FSDP, tensor or pipeline parallelism, gradient
  compression or custom collective kernels.
- **Rejected:** the original README plan, a from-scratch framework.
- **Cost:** no "I built a framework" story. In exchange, the depth goes where it is
  rare: NCCL failure semantics.

## ADR-2: Keep a thin C++ layer on the raw NCCL API
- **Decision:** the communicator lives in C++ with pybind11 bindings and uses
  `ncclCommInitRankConfig` (non-blocking), `ncclCommGetAsyncError`, `ncclCommAbort`,
  `ncclCommFinalize` and `ncclCommSplit` directly.
- **Rejected:** pure Python on `torch.distributed`, which hides exactly the behaviour
  under study.
- **Cost:** a CUDA/NCCL build to maintain.

## ADR-3: This repo is the NCCL layer; goodput is the orchestrator
- **Decision:** dtf is scoped to the fault-aware communicator and the failure
  experiments. Detection, injection, classification, checkpointing, restart and goodput
  accounting live in
  [goodput](https://github.com/nareshns2004/goodput-fault-tolerant-training-orchestrator).
- **Rejected:** (a) re-scoping dtf into a second orchestrator, as its master booklet
  first proposed, which would duplicate goodput; (b) archiving dtf, which would discard
  the only place where NCCL failure behaviour is studied directly.
- **Cost:** a second, small repo to keep coherent with goodput.

## ADR-4: Every error aborts every communicator in the process
- **Decision:** a process-wide registry; any async error, deadline expiry or failed call
  triggers `abort_all()`. `ncclCommDestroy` is used only on healthy communicators,
  after `ncclCommFinalize`.
- **Rejected:** per-communicator handling, which leaves survivors hung on the other
  groups; unconditional destroy, which can block forever after a peer has died.
- **Cost:** one bad sub-group takes down the whole process's NCCL state. That is the
  correct trade-off for synchronous training.

## ADR-5: Deadlines on every blocking wait
- **Decision:** init and collective completion waits take a configurable deadline;
  expiry is an error that goes through ADR-4.
- **Rejected:** blocking mode with an external timeout (cannot cancel a blocked init).
- **Cost:** polling loops and `ncclInProgress` handling throughout. Deadline values are
  configuration, not constants; goodput derives them from measured step time.

## ADR-6: The findings are the deliverable; the library is optional for goodput
- **Decision:** each failure experiment (design §5) ships as a GPU test plus a written
  finding. goodput consumes the findings for its M2 spike and may use `dtf_C` from its
  recovery hooks, but has no hard dependency on it.
- **Rejected:** making goodput depend on dtf, which conflicts with goodput's ADR-0002
  ("C++ only where it earns it") until measurements show a need.
- **Cost:** some behaviour gets checked twice, once on raw NCCL and once through
  `ProcessGroupNCCL`.

## ADR-7: No hand-written numbers or feature claims
- **Decision:** every number in README or docs comes from a committed script and names
  GPU, driver, CUDA, NCCL and PyTorch versions. Unbuilt features are labelled "planned".
- **Rejected:** hand-written tables, which is what the original README had.
- **Cost:** slower to publish anything.

## ADR-8: Build timing follows goodput
- **Decision:** after M1/M1b (cleanup and design), dtf's build milestones (M2, M3) start
  when goodput starts its M2, because that is where their output is used.
- **Rejected:** building dtf now as a third parallel track beside goodput and kvwire.
- **Cost:** the README stays "design only" until then, which is honest and easy to
  explain.
