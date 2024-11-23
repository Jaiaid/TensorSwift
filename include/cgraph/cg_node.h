#ifndef _CG_NODE_H
#define _CG_NODE_H

#include <string>

#include <core/ts.h>

namespace cgraph
{
    class CGNode
    {
    public:
        std::string name;
        ts::SwiftTensor parameter;
        ts::SwiftTensor gradient;

        CGNode()
        {
            this->name = "null";
        }

        CGNode(std::string name)
        {
            this->name = name;
        }

        CGNode(std::string name, ts::SwiftTensor weight)
        {
            this->name = name;
        }
    };

    namespace cgnode_util
    {
        void random_init(CGNode& parameter);
    }
}
#endif