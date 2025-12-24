#include "dtf/communicator.h"
#include "dtf/utils.h"

namespace dtf {

Communicator::Communicator(int rank, int world_size) 
    : rank_(rank), world_size_(world_size) {}

Communicator::~Communicator() {
    if (comm_) {
        ncclCommDestroy(comm_);
    }
}

void Communicator::init(const ncclUniqueId& nccl_id) {
    init_rank(rank_, world_size_, nccl_id);
}

void Communicator::init_rank(int rank, int world_size, const ncclUniqueId& nccl_id) {
    rank_ = rank;
    world_size_ = world_size;
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size_, nccl_id, rank_));
}

ncclUniqueId Communicator::get_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return id;
}

} // namespace dtf
