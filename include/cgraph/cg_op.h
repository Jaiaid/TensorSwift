#ifndef _CG_OP_H
#define _CG_OP_H

#include <vector>
#include <string>

#include <core/ts.h>
#include <cgraph/cg_node.h>

namespace cgraph
{
    class CGOp
    {
    protected:
        std::string name;
        std::vector<CGNode> inputs;
        CGNode output;

    public:
        CGOp();
        CGNode& get_activation_ref();
        void forward(const CGNode& external_input);
        virtual void forward(const ts::SwiftTensor& external_input);
        virtual void backward();
    };
}
#endif