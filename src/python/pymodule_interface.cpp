#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <core/python/core_interface.h>
#include <cgraph/python/cgraph_interface.h>
#include <python/pymodule_interface.h>

PYBIND11_MODULE(MODULE_NAME, m)
{
    m.doc() = "TensorSwift plugin by pybind11";

    init_core_interface(m);
    init_cgraph_interface(m);
}