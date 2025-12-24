#pragma once

#include <stdexcept>
#include <string>

namespace dtf {

#define DTF_CHECK(cmd) do { \
    if (!(cmd)) { \
        throw std::runtime_error(std::string("DTF check failed: ") + #cmd); \
    } \
} while(0)

#define NCCL_CHECK(cmd) do { \
    ncclResult_t res = cmd; \
    if (res != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(res)); \
    } \
} while(0)

} // namespace dtf
