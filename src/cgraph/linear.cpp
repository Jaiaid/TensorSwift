#include <cgraph/linear.h>
#include <cgraph/cg_node.h>

Linear::Linear(int num_neuron, int num_input, bool bias):CGOp()
{
    num_input = bias?num_input + 1:num_input;
    this->inputs = std::vector<cgraph::CGNode>();
    this->inputs.push_back(cgraph::CGNode("linear", ts::SwiftTensor({num_input, num_neuron})));
}

/**
 * Expected shape of input is (batch size, feature)
 * If bias is true
 */
void Linear::forward(const ts::SwiftTensor &external_input)
{
    if (external_input.shape[1] != this->inputs[0].parameter.shape[0]) {
        throw std::invalid_argument("number of feature in input not matching number of neuron input");
    }
    this->output = cgraph::CGNode("linear activation", this->inputs[1].parameter.matmul(this->inputs[0].parameter));
}

void Linear::backward()
{
}