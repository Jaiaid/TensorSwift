#include <cgraph/cg_node.h>

cgraph::CGOp::CGOp()
{
}

cgraph::CGNode &cgraph::CGOp::get_activation_ref()
{
    // TODO: insert return statement here
    return this->output;
}

void cgraph::CGOp::forward(const CGNode &external_input)
{
    this->forward(external_input.parameter);
}

void cgraph::CGOp::forward(const ts::SwiftTensor &external_input)
{
}

void cgraph::CGOp::backward()
{
}
