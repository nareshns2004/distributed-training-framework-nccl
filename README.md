# distributed-training-framework-nccl

A high-performance distributed deep learning training framework built on NCCL, supporting multi-node, multi-GPU training with Python bindings and CUDA kernels.

## Features

- Multi-node, multi-GPU training via NCCL collectives (AllReduce, AllGather, Broadcast, ReduceScatter)
- Custom CUDA kernels for fused operations (gradient compression, mixed-precision casting)
- Python API compatible with PyTorch and JAX workflows
- Gradient compression and sparsification to reduce communication overhead
- Fault-tolerant training with checkpoint and resume support
- Support for data parallelism, tensor parallelism, and pipeline parallelism

## Requirements

- CUDA 11.8+ and cuDNN 8.6+
- NCCL 2.16+
- Python 3.9+
- PyTorch 2.0+ or JAX 0.4+
- CMake 3.20+
- GCC 10+ or Clang 13+
- OpenMPI 4.1+ (optional, for MPI backend)

## Installation

### From source

```bash
git clone https://github.com/your-org/distributed-training-framework-nccl.git
cd distributed-training-framework-nccl

# Build C++/CUDA extensions
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DNCCL_ROOT=/path/to/nccl -DCUDA_ARCH=80
make -j$(nproc)
cd ..

# Install Python package
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t dtf-nccl:latest .
docker run --gpus all --rm -it dtf-nccl:latest
```

## Quick Start

```python
import dtf

# Initialize the process group
dtf.init(backend="nccl", world_size=8)

# Wrap your model
model = dtf.DistributedModel(
    my_model,
    parallelism="data",
    gradient_compression=True,
)

# Training loop works like normal
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

For multi-node launch:

```bash
dtf-launch --nnodes 4 --nproc-per-node 8 --master-addr 10.0.0.1 train.py
```

## Documentation

Full documentation is available at `docs/` or online at [your-docs-url].

- [Architecture overview](docs/architecture.md)
- [API reference](docs/api.md)
- [Configuration guide](docs/configuration.md)
- [Benchmarks](docs/benchmarks.md)

## Benchmarks

| Model        | GPUs | Nodes | Throughput      | Scaling efficiency |
|--------------|------|-------|-----------------|-------------------|
| GPT-2 (1.5B) | 8    | 1     | 142 samples/sec | 94%               |
| GPT-2 (1.5B) | 64   | 8     | 1,090 samples/sec | 91%             |
| LLaMA-7B     | 64   | 8     | 38 samples/sec  | 88%               |

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
