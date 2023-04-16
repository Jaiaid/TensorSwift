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
        .def(pybind11::init<const std::vector<int>&>())
        .def(pybind11::init<const std::vector<float>&, const std::vector<int>&>())
        .def("view", &SwiftTensor::view, "get changed view of SwiftTensor with same storage")
        .def("size", &SwiftTensor::size, "get total number of element in SwiftTensor")
        .def_readonly("shape", &SwiftTensor::shape, "get shape of SwiftTensor")
        .def(pybind11::self + pybind11::self)
        .def(pybind11::self - pybind11::self)
        .def(pybind11::self * pybind11::self)
        .def(pybind11::self / pybind11::self)
        .def(pybind11::self + float())
        .def(pybind11::self - float())
        .def(pybind11::self * float())
        .def(pybind11::self / float())
        .def(float() + pybind11::self)
        .def(float() - pybind11::self)
        .def(float() * pybind11::self)
        .def("multiply", &SwiftTensor::multiply, "multiply two objects of class SwiftTensor")
        .def("dot", &SwiftTensor::dot, "multiply and sum two 1D objects of class SwiftTensor")
        .def("matmul", &SwiftTensor::matmul, "multiply and sum two 2D objects of class SwiftTensor")
        .def("sum", &SwiftTensor::sum, "sum the data of a SwiftTensor")
        .def("__getitem__", static_cast<float (SwiftTensor::*)(int)const>(&SwiftTensor::operator[]))
        .def("__getitem__", static_cast<float (SwiftTensor::*)(const::std::vector<int>&)const>(&SwiftTensor::operator[]))
        .def("__setitem__", static_cast<void (SwiftTensor::*)(int, float val)const>(&SwiftTensor::set))
        .def("__setitem__", static_cast<void (SwiftTensor::*)(const::std::vector<int>&, float val)const>(&SwiftTensor::set))
        .def("__repr__", &tensorswift_stringify);
}