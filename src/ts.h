#ifndef _TS_H
#define _TS_H

#include <memory>
#include <iostream>

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
        // calculate the size
        uint64_t intended_size = (shape.size()>0)?shape[0]:0;
        for(uint64_t i=1;i<shape.size();i++)
        {
            intended_size*=shape[i];
        }
        
        // create storage
        this->storage_ptr = std::make_shared<Storage>(intended_size);
        this->shape = shape;
    }

    SwiftTensor(std::shared_ptr<Storage> storage_ptr, std::vector<int>& new_shape)
    {
        // calculate the size corresponding to new shape
        uint64_t intended_size = (new_shape.size()>0)?new_shape[0]:0;
        for(uint64_t i=1;i<new_shape.size();i++)
        {
            intended_size*=new_shape[i];
        }

        // if size donot match just return an empty tensor with empty storage
        if (intended_size != storage_ptr->size) {
            this->storage_ptr = std::make_shared<Storage>();
            this->shape = std::vector<int>();
        }
        // else assign the same storage and new shape
        else {
            this->storage_ptr = storage_ptr;
            this->shape = new_shape;
        }
    }
    
    // return a new instance with changed view but with same storage
    SwiftTensor view(std::vector<int>& shape)
    {
        SwiftTensor new_view = SwiftTensor(this->storage_ptr, shape);
        return new_view;
    }

    // return total number of element
    int size()
    {
        return this->storage_ptr->size;
    }

    SwiftTensor operator+(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator-(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator*(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator/(const SwiftTensor& t)const { return SwiftTensor(); }
};

#endif
