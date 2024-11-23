#ifndef _PYTHON_INTERFACE_TENSORSWIFT_H
#define _PYTHON_INTERFACE_TENSORSWIFT_H

#include <pybind11/pybind11.h>

#define MODULE_TENSORCLASS_PYTHONNAME "SwiftTensor"
#define MODULE_GENCONTAINER_PYTHONNAME "Data"

void init_core_interface(pybind11::module &m);

#endif