#include <cuda_runtime.h>
#include <nccl.h>

// Custom AllReduce logic can go here in the future
// Currently deferring to NCCL's built-in ncclAllReduce in collective_ops.cpp
