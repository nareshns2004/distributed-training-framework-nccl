#include "dtf/collective_ops.h"
#include "dtf/utils.h"

namespace dtf {

void CollectiveOps::all_reduce(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op,
    std::shared_ptr<Communicator> comm, cudaStream_t stream) {
    
    NCCL_CHECK(ncclAllReduce(
        sendbuff, recvbuff, count, datatype, op, 
        comm->get_nccl_comm(), stream));
}

void CollectiveOps::broadcast(
    const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root,
    std::shared_ptr<Communicator> comm, cudaStream_t stream) {
    
    NCCL_CHECK(ncclBroadcast(
        sendbuff, recvbuff, count, datatype, root, 
        comm->get_nccl_comm(), stream));
}

void CollectiveOps::all_gather(
    const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype,
    std::shared_ptr<Communicator> comm, cudaStream_t stream) {
    
    NCCL_CHECK(ncclAllGather(
        sendbuff, recvbuff, sendcount, datatype, 
        comm->get_nccl_comm(), stream));
}

} // namespace dtf
