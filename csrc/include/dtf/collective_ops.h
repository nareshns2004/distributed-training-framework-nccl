#pragma once

#include "dtf/communicator.h"
#include <nccl.h>
#include <cuda_runtime.h>

namespace dtf {

class CollectiveOps {
public:
    static void all_reduce(
        const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, ncclRedOp_t op,
        std::shared_ptr<Communicator> comm, cudaStream_t stream);

    static void broadcast(
        const void* sendbuff, void* recvbuff, size_t count,
        ncclDataType_t datatype, int root,
        std::shared_ptr<Communicator> comm, cudaStream_t stream);
        
    static void all_gather(
        const void* sendbuff, void* recvbuff, size_t sendcount,
        ncclDataType_t datatype,
        std::shared_ptr<Communicator> comm, cudaStream_t stream);
};

} // namespace dtf
