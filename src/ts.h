#ifndef _TS_H
#define _TS_H

#include <memory>
#include "storage.h"

class SwiftTensor
{
    std::shared_ptr<Storage> storage_ptr;
    std::vector<int> dim_offset;
public:
    std::vector<int> shape;
    
    SwiftTensor(){ this->storage_ptr = std::make_shared<Storage>(); }
    SwiftTensor(std::vector<int>& shape)
    { 
        this->storage_ptr = std::make_shared<Storage>();
        this->shape = shape;
    }
    SwiftTensor(std::shared_ptr<Storage> storage_ptr, std::vector<int>& shape)
    { 
        this->storage_ptr = storage_ptr;
        this->shape = shape;
    }
    
    SwiftTensor view(std::vector<int>& shape)
    {
        SwiftTensor new_view = SwiftTensor(this->storage_ptr, shape);
        return new_view;
    }

    SwiftTensor operator+(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator-(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator*(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator/(const SwiftTensor& t)const { return SwiftTensor(); }
};

#endif
