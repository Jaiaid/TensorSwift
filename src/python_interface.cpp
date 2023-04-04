#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "python_interface.h"
#include "data.h"
#include "ts.h"

PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "TensorSwift plugin by pybind11";

    pybind11::class_<Data>(m, MODULE_GENCONTAINER_PYTHONNAME)
        .def(pybind11::init<>())
        .def(pybind11::init<int>())
        .def("get", &Data::get, "get internal value of Data class")
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self - pybind11::self)
        .def(pybind11::self * pybind11::self)
        .def(pybind11::self / pybind11::self)
        .def(int() + pybind11::self)
        .def(int() - pybind11::self)
        .def(int() * pybind11::self)
        .def(int() / pybind11::self)
        .def("__repr__",
            [](const Data& d)
            {
                return std::to_string(d.get());
            }
        );

    pybind11::class_<SwiftTensor>(m, MODULE_TENSORCLASS_PYTHONNAME)
        .def(pybind11::init<>())
        .def(pybind11::init<std::vector<int>&>())
        .def("view", &SwiftTensor::view, "get changed view of SwiftTensor with same storage")
        .def_readonly("shape", &SwiftTensor::shape, "get shape of SwiftTensor")
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self - pybind11::self)
        .def(pybind11::self * pybind11::self)
        .def(pybind11::self / pybind11::self)
        .def("__repr__",
            [](const SwiftTensor& d)
            {
                return "";
            }
        );
}