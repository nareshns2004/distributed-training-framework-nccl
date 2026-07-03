#pragma once

#include <nccl.h>
#include <vector>
#include <memory>

namespace dtf {

class Communicator {
public:
    Communicator(int rank, int world_size);
    ~Communicator();

    void init(const ncclUniqueId& nccl_id);
    void init_rank(int rank, int world_size, const ncclUniqueId& nccl_id);

    ncclComm_t get_nccl_comm() const { return comm_; }
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }

    static ncclUniqueId get_unique_id();

private:
    ncclComm_t comm_ = nullptr;
    int rank_;
    int world_size_;
};

} // namespace dtf
