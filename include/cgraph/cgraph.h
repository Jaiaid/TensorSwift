#ifndef _CGRAPH_H
#define _CGRAPH_H

#include <map>

#include <cgraph/cg_node.h>
#include <cgraph/cg_op.h>

namespace cgraph
{
    class CGraph
    {
        std::vector<CGOp> opsequence;

    public:
        std::vector<CGNode> parameterlist;
        
        CGraph(){};

        std::vector<CGOp>& sequence()
        {
            return this->opsequence;
        }

        void compute(const ts::SwiftTensor& input)
        {
            this->opsequence[0].forward(input);
            CGNode& activation = this->opsequence[0].get_activation_ref();

            for(int i = 1;i < opsequence.size();i++)
            {
                this->opsequence[0].forward(activation);
                CGNode& activation = this->opsequence[0].get_activation_ref();
            }
        }

        const CGNode& output()
        {
            return this->opsequence[opsequence.size() - 1].get_activation_ref();
        }
    };
}
#endif