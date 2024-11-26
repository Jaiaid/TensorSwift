#ifndef _LINEAR_OP_H_
#define _LINEAR_OP_H_

#include <core/ts.h>
#include <cgraph/cg_op.h>

class Linear:public cgraph::CGOp
{
public:
    Linear(int num_neuron, int num_input, bool bias=true);
    void forward(const ts::SwiftTensor& external_input);
    void backward();
};

#endif