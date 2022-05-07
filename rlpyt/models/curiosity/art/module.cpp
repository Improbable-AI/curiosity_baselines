#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "OnlineFuzzyArt.h"

using namespace art;

namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(std::vector<double>);
// PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>);

PYBIND11_MODULE(art, m)
{

    // // m.doc() = "pybind11 example plugin"; // optional module docstring
    py::class_<OnlineFuzzyART>(m, "OnlineFuzzyART")
        .def(py::init<double, double, double, int>())
        .def("run_online", &OnlineFuzzyART::run_online, py::arg("features"), py::arg("max_epochs") = std::numeric_limits<int>::max());

        // std::vector<int> OnlineFuzzyART::run_online(const MatrixConstRef features, int max_epochs)

    // m.def("test_optional", &test_optional);
    // // m.def("test_shared_ptr", &test_shared_ptr);
    // m.def("test_ptr", &test_ptr);
    // m.def("test_vec", &test_vec);


    // // .def("setName", &Pet::setName)
    // // .def("getName", &Pet::getName);
    // py::module_ targets = m.def_submodule("targets");
    // py::class_<Target>(targets, "Target")
    //     .def(py::init<const VectorConstRef, std::shared_ptr<DynamicModel>, int>(), py::arg("init_state"), py::arg("dyn_model"), py::arg("len_story") = 1);

    // py::module_ dynamics = m.def_submodule("dynamics");
    // py::class_<DynamicModel, std::shared_ptr<DynamicModel>, PyDynamicModel>(dynamics, "DynamicModel")
    //     .def(py::init<>())
    //     .def("state_space_dim", &DynamicModel::state_space_dim)
    //     .def("pos_idx", &DynamicModel::pos_idx)
    //     .def("input_vec", &DynamicModel::input_vec)
    //     .def("gain_mat", &DynamicModel::gain_mat)
    //     .def("cov_mat", &DynamicModel::cov_mat)
    //     .def("dyn_mat", &DynamicModel::dyn_mat)
    //     .def("propagate", &DynamicModel::propagate);
}