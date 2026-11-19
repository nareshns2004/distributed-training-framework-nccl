# dtf design: a fault-aware NCCL layer

Status: accepted design, nothing built yet (19 Nov 2026). Decisions are recorded in
[`design-decisions.md`](design-decisions.md).

## 1. Problem

In NCCL-based training, a dead or hung peer doesn't produce an error. NCCL calls are
asynchronous: `ncclAllReduce` returns once the work is enqueued, so the survivors'
kernels just never finish. Recovering from that takes NCCL APIs most engineers never
touch: non-blocking init with a deadline, async error polling, `ncclCommAbort` and
communicator re-creation. PyTorch's `ProcessGroupNCCL` uses these internally but hides
which operations are legal after a failure, how long each step takes, and when the CUDA
context itself is poisoned.

Those unknowns block in-process recovery. The
[goodput](https://github.com/nareshns2004/goodput-fault-tolerant-training-orchestrator)
orchestrator lists them as open questions for its M2 in-process restart feasibility
spike (`goodput/rank/recovery.py`):

- What state is a communicator in after an async error or an abort, and which
  operations are still legal on it?
- When is the CUDA context poisoned (sticky errors) so that the process must exit?
- What does it cost to abort and re-create communicators for a new world?

**dtf answers these on raw NCCL, with code you can read in an afternoon.** It
provides a minimal C++ communicator that is correct under failure, a test suite that
reproduces each failure, and written findings with measured timings.

## 2. Scope

In scope:

- A C++ `Communicator` built on `ncclCommInitRankConfig` with `blocking = 0`: init
  and collectives that take a deadline, `ncclInProgress` handled everywhere.
- Async error polling (`ncclCommGetAsyncError`) on a background thread, with a callback.
- A process-wide registry so that one error aborts **every** communicator the process
  holds (otherwise survivors hang on the other groups).
- Safe teardown: `ncclCommFinalize` + `ncclCommDestroy` when healthy, `ncclCommAbort`
  otherwise. `ncclCommSplit` for sub-groups.
- Collectives (all-reduce, all-gather, reduce-scatter, broadcast) bound to Python,
  taking device pointers and a CUDA stream handle from PyTorch.
- Failure experiments and written findings (section 5) plus the benchmarks behind
  them (section 6).

Not in scope, because goodput owns it: watchdog and hang detection, heartbeats, the
fault injector (goodput's `faultlab`), the failure classifier, checkpointing, the
orchestrator and launcher, goodput accounting, and nicprof integration.

Not in scope at all, because PyTorch and NCCL already do it well: DDP/FSDP, tensor and
pipeline parallelism, gradient compression, custom all-reduce or cast kernels, JAX,
and an MPI backend.

## 3. Architecture

```
 Python (tests, experiments, optionally goodput's RecoveryHooks)
   │  dtf_C: get_unique_id · Communicator · collectives(ptr, count, dtype, stream)
   ▼
 ┌──────────────────────────── dtf_C (C++17) ─────────────────────────────┐
 │ CommRegistry (process-wide)     abort_all() · on_error(callback)        │
 │   └─ Communicator               state: Initializing → Healthy           │
 │        create(id, rank, n,                 → Errored → Aborted / Closed │
 │               deadline)        poll() · abort() · close() · split()     │
 │   └─ ErrorPoller thread        ncclCommGetAsyncError on every comm;     │
 │                                 on error: abort_all() then callback     │
 │ Collectives                    enqueue → wait_ncclInProgress(deadline); │
 │                                 per-communicator sequence counter       │
 └──────────────────────────────── NCCL ──────────────────────────────────┘
```

Rules the code enforces (they are also in `CLAUDE.md`):

1. A communicator that has seen an error is never passed to `ncclCommDestroy`; it is
   aborted.
2. Any error on any communicator aborts all communicators in the process.
3. Every blocking wait has a deadline. A deadline expiring counts as an error.
4. Error messages carry rank, world size, operation and NCCL error string.

How the bootstrap works: rank 0 calls `ncclGetUniqueId`; the 128-byte ID travels
over a c10d `TCPStore` (the same store goodput uses), never over NCCL itself.

## 4. Python surface

Kept deliberately small. It isn't a training API.

```python
import dtf_C
uid = dtf_C.get_unique_id()                       # rank 0; share via TCPStore
comm = dtf_C.Communicator.create(uid, rank, world, init_deadline_ms=30_000)
comm.all_reduce(t.data_ptr(), t.numel(), dtype, op, torch.cuda.current_stream().cuda_stream)
comm.poll()        # -> "ok" | NCCL error name
dtf_C.abort_all()  # registry-wide
```

goodput may call this from its `abort_and_reinit_comms()` hook. It isn't required to:
the findings in section 5 are the main deliverable to goodput, and the library is
optional (ADR-6).

## 5. Failure experiments → `docs/findings/`

Each experiment is a GPU test that reproduces the situation, plus a short write-up of
the observed behaviour, the NCCL/driver/PyTorch versions and the timings measured by
the test. Check version-specific NCCL behaviour against the NCCL docs for the version
under test.

| ID | Situation | Question answered |
|---|---|---|
| X1 | One rank never calls init | Do the others leave cleanly at the deadline? Is the process reusable afterwards? |
| X2 | A peer is SIGKILLed mid all-reduce | What does a survivor observe (async error or silent hang), and after how long? |
| X3 | A peer is SIGSTOPped mid all-reduce | Same as X2 for a hang rather than a crash. |
| X4 | `ncclCommDestroy` after a peer died | Reproduce the hang; confirm that `ncclCommAbort` returns, and how fast. |
| X5 | Abort, then re-create in the same process | Can the survivors build a new communicator (new ID, smaller world) without restarting? What does it cost? |
| X6 | Error in one of several communicators | Do the other groups hang unless `abort_all` runs? |
| X7 | CUDA sticky error (e.g. illegal address in a kernel) | Is the context unrecoverable, so the process must exit? |
| X8 | `ncclCommSplit` after a parent error | Is split legal, and what state do the children inherit? |

X2, X5 and X7 together answer goodput's M2 spike questions.

## 6. Measurements

Every number is generated by a script under `bench/` and written to
`bench/RESULTS.md` with GPU model, driver, CUDA, NCCL and PyTorch versions.

| ID | What | How |
|---|---|---|
| E1 | NCCL baseline busbw | `all_reduce_perf -b 8 -e 1G -f 2 -g <n>` (nccl-tests) |
| E2 | Wrapper overhead | Same sizes through `dtf_C` versus nccl-tests; expect ≈ 0 |
| E3 | Init time vs world size | Non-blocking init, 2 → 8 GPUs, 10 trials |
| E4 | Abort latency | Time for `abort()` to return after a peer has died (X2/X4) |
| E5 | Re-create cost | abort → new ID exchanged → new communicator healthy (X5); feeds goodput's MTTR budget |
| E6 | Poller overhead | Step time of a fixed all-reduce loop with the poller on vs off |

Hardware: one host with 2 or more GPUs covers X1–X8 and E1–E6. A network-partition
variant of X2 needs two hosts with RDMA NICs and comes later.

## 7. Milestones

| Milestone | Content | Booklet prompt | Gate |
|---|---|---|---|
| M1 | Honest README, hygiene, CLAUDE.md | P0 | done |
| M1b | This design and the ADRs | P1 | done |
| M2 | Modern CMake, CI (ruff, mypy, CPU tests, CMake configure) | P2 | starts with goodput's M2 |
| M3 | Fault-aware communicator, bindings, GPU tests, X1–X8 findings, E1–E6 | P3 | feeds goodput's M2 spike write-up |

Booklet prompts P4–P9 (workload, checkpointing, watchdog, injector, orchestrator,
recovery benchmarks, nicprof) are goodput's work and are tracked in its milestones.
