# distributed-training-framework-nccl

# Mini Distributed Training Framework using NCCL

A minimal, educational distributed training framework built from scratch using **NCCL**, **CUDA/C++**, and a thin **Python API**. This project is designed for engineers who want to understand the *systems and infrastructure* behind modern large-scale AI training—without hiding behind heavyweight frameworks.

This is **not** a PyTorch clone. It is a learning framework focused on:

* GPU communication
* Distributed systems fundamentals
* Training system internals
* AI infrastructure–grade engineering

---

## Why This Project

Modern AI training stacks (PyTorch, JAX, DeepSpeed, Megatron, etc.) abstract away critical infrastructure layers. This project rebuilds those layers in a minimal form to understand:

* How NCCL collectives actually work
* How data-parallel training is implemented
* How gradients move across GPUs
* How training systems are architected end-to-end

If your goal is to work in **AI Infrastructure, Systems, or ML Platforms**, this project builds the right mental model.

---

## Features (Planned)

* NCCL-based GPU collectives

  * allreduce, broadcast, barrier
* Multi-process launcher (1 process per GPU)
* Minimal CUDA Tensor abstraction
* Lightweight autograd for demo models
* Distributed SGD/Adam
* Python training API
* Example: data-parallel MNIST training

---

## Architecture Overview

### High-Level Flow

```
User Script (Python)
        |
        v
  Trainer API
        |
        v
Process Group (NCCL Wrapper)
        |
        v
   CUDA + NCCL Backend
```

### Core Components

1. **Launcher**

   * Spawns N processes (one per GPU)
   * Sets env vars: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`

2. **Process Group**

   * Wraps NCCL communicators
   * Exposes: `allreduce`, `broadcast`, `barrier`

3. **Tensor Layer**

   * CUDA memory management
   * Basic ops (add, mul, matmul later)

4. **Autograd Lite**

   * Manual backward graph
   * Enough to train simple models

5. **Optimizer**

   * SGD, Adam
   * Uses NCCL allreduce for gradients

---

## Repository Structure

```
mini-ddp-nccl/
│
├── launcher/
│   └── launch.py           # Multi-process launcher
│
├── cpp/
│   ├── nccl_comm.cpp       # NCCL wrapper
│   ├── nccl_comm.h
│   ├── tensor.cu           # CUDA tensor ops
│   ├── tensor.h
│   └── bindings.cpp        # pybind11 bindings
│
├── python/
│   ├── process_group.py    # Python NCCL API
│   ├── tensor.py           # Python Tensor wrapper
│   ├── optimizer.py
│   └── trainer.py
│
├── examples/
│   └── mnist_ddp.py        # Distributed training example
│
├── CMakeLists.txt
└── README.md
```

---

## Build Roadmap

### Phase 1 — NCCL Hello World

* Initialize NCCL communicator
* Run allreduce across GPUs
* Verify correctness

### Phase 2 — C++ NCCL Wrapper

* Clean C++ abstraction over NCCL
* Expose via pybind11

### Phase 3 — Tensor Layer

* CUDA memory allocation
* Host ↔ device copy
* Basic ops

### Phase 4 — Autograd Lite

* Minimal backward graph
* Linear model support

### Phase 5 — Distributed Training

* Gradient allreduce
* Data-parallel training loop

---

## First Milestone

**Goal:**

* Run on 2+ GPUs
* Each rank prints identical allreduced result

This validates:

* NCCL setup
* Process launching
* GPU communication

---

## Requirements

* Linux
* NVIDIA GPU(s)
* CUDA Toolkit
* NCCL
* CMake
* Python 3.9+

Optional but recommended:

* `nccl-tests` for validation

---

## How This Helps Your AI Infra Career

This project demonstrates skills directly aligned with:

* GPU communication systems
* Distributed training platforms
* ML infrastructure engineering
* FAANG+/AI Lab–style systems work

It shows you can work below the framework layer—where real scaling problems live.

---

## License

MIT
