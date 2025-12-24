#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dtf/communicator.h"
#include "dtf/process_group.h"
#include "dtf/collective_ops.h"
#include "dtf/utils.h"

namespace py = pybind11;

PYBIND11_MODULE(dtf_C, m) {
    m.doc() = "dtf C++/CUDA backend bindings";

    // Expose ncclUniqueId as a bytes object for easy Python serialization
    m.def("get_unique_id", []() {
        ncclUniqueId id = dtf::Communicator::get_unique_id();
        return py::bytes(id.internal, NCCL_UNIQUE_ID_BYTES);
    }, "Get a new NCCL unique ID");

    py::class_<dtf::Communicator, std::shared_ptr<dtf::Communicator>>(m, "Communicator")
        .def(py::init<int, int>())
        .def("get_rank", &dtf::Communicator::get_rank)
        .def("get_world_size", &dtf::Communicator::get_world_size);

    py::class_<dtf::ProcessGroup, std::shared_ptr<dtf::ProcessGroup>>(m, "ProcessGroup")
        .def(py::init<int, int>())
        .def("init", [](dtf::ProcessGroup& pg, py::bytes id_bytes) {
            std::string str = id_bytes;
            if (str.size() != NCCL_UNIQUE_ID_BYTES) {
                throw std::runtime_error("Invalid NCCL ID size");
            }
            ncclUniqueId id;
            std::memcpy(id.internal, str.data(), NCCL_UNIQUE_ID_BYTES);
            pg.init(id);
        })
        .def("get_communicator", &dtf::ProcessGroup::get_communicator)
        .def("get_rank", &dtf::ProcessGroup::get_rank)
        .def("get_world_size", &dtf::ProcessGroup::get_world_size);
}
