#include "dtf/process_group.h"
#include "dtf/utils.h"

namespace dtf {

ProcessGroup::ProcessGroup(int rank, int world_size) 
    : rank_(rank), world_size_(world_size) {
    comm_ = std::make_shared<Communicator>(rank, world_size);
}

ProcessGroup::~ProcessGroup() {}

void ProcessGroup::init(const ncclUniqueId& nccl_id) {
    comm_->init(nccl_id);
}

} // namespace dtf
