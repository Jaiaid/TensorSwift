#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <core/ts.h>
#include <cgraph/cgraph.h>
#include <cgraph/cg_op.h>
#include <cgraph/cg_node.h>
#include <cgraph/linear.h>
#include <cgraph/python/cgraph_interface.h>

void init_cgraph_interface(pybind11::module &m)
{
    pybind11::class_<cgraph::CGNode>(m, MODULE_CGRAPHNODECLASS_PYTHONNAME)
        .def(pybind11::init<>())
        .def(pybind11::init<std::string>())
        .def(pybind11::init<std::string, ts::SwiftTensor&>())
        .def_readonly("name", &cgraph::CGNode::name, "get name of computation node")
        .def_readonly("parameter", &cgraph::CGNode::parameter, "get parameter associated with computation node")
        .def_readonly("gradient", &cgraph::CGNode::gradient, "get gradient associated with computation node parameter");

    pybind11::class_<cgraph::CGOp>(m, MODULE_CGRAPHOPCLASS_PYTHONNAME)
        .def(pybind11::init<>())
        .def("forward", static_cast<void (cgraph::CGOp::*)(const ts::SwiftTensor&)>(&cgraph::CGOp::forward), "apply the operation using the argument as input")
        .def("forward", static_cast<void (cgraph::CGOp::*)(const cgraph::CGNode&)>(&cgraph::CGOp::forward), "apply the operation using the argument as input")
        .def("backward", &cgraph::CGOp::backward, "calculate gradient of output w.r.t parameter");

    pybind11::class_<cgraph::CGraph>(m, MODULE_CGRAPHCLASS_PYTHONNAME)
        .def(pybind11::init<>())
        .def_readonly("parameters", &cgraph::CGraph::parameterlist, "get parameters of the computation graph")
        .def("add", static_cast<void (cgraph::CGraph::*)(cgraph::CGOp *)>(&cgraph::CGraph::add_op), "append a new operation to the graph")
        .def("sequence", static_cast<const std::vector<cgraph::CGOp *>&(cgraph::CGraph::*)()>(&cgraph::CGraph::sequence), "get the sequence of operation as a vector")
        .def("compute", static_cast<void (cgraph::CGraph::*)(const ts::SwiftTensor&)>(&cgraph::CGraph::compute), "compute the chain based on input")
        .def("output", static_cast<const cgraph::CGNode& (cgraph::CGraph::*)()>(&cgraph::CGraph::output), "output the result");
        

    pybind11::class_<Linear>(m, "Linear")
        .def(pybind11::init<int, int, bool>())
        .def("forward", static_cast<void (Linear::*)(const ts::SwiftTensor&)>(&Linear::forward), "apply the operation using the argument as input")
        .def("backward", &Linear::backward, "calculate gradient of output w.r.t parameter");

}