#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sofm/art/OnlineFuzzyArt.h"

using namespace sofm::art;

namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(std::vector<double>);

PYBIND11_MODULE(sofm, m)
{
    py::module_ m_art = m.def_submodule("art");
    py::class_<OnlineFuzzyART>(m_art, "OnlineFuzzyART")
        .def(py::init<double, double, double, int>(), py::arg("rho"), py::arg("alpha"), py::arg("beta"), py::arg("num_features"))
        .def("run_online", &OnlineFuzzyART::run_online, py::arg("features"), py::arg("max_epochs") = std::numeric_limits<int>::max());
}