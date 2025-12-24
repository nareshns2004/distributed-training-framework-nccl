#pragma once

#include "dtf/communicator.h"
#include <memory>

namespace dtf {

class ProcessGroup {
public:
    ProcessGroup(int rank, int world_size);
    ~ProcessGroup();

    void init(const ncclUniqueId& nccl_id);
    std::shared_ptr<Communicator> get_communicator() const { return comm_; }

    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }

private:
    int rank_;
    int world_size_;
    std::shared_ptr<Communicator> comm_;
};

} // namespace dtf
