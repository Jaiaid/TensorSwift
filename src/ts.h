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
    
    SwiftTensor(const std::vector<int>& shape)
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

    SwiftTensor(std::shared_ptr<Storage> storage_ptr, const std::vector<int>& new_shape)
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
    SwiftTensor view(const std::vector<int>& shape)
    {
        SwiftTensor new_view = SwiftTensor(this->storage_ptr, shape);
        return new_view;
    }

    // return total number of element
    int size()
    {
        return this->storage_ptr->size;
    }

    // get the storage buffer to read
    const Storage& get_storage()
    {
        return *(this->storage_ptr);
    }

    // element wise addition
    // considers the underlying buffer as flattened array and add corresponding element
    // works only for tesnor with same size (shape may be different)
    SwiftTensor operator+(const SwiftTensor& t)const 
    {
        // we can add stuffs considering the buffer as 1d for + operation
        // what ever the shape, as long as size matches this should work
        // for axis specific addition, it will be done in separate function to pass axis parameter

        // TODO
        // size check and throw exception
        // generate a tensor with independent storage but same shape
        SwiftTensor result = SwiftTensor(this->shape);

        return SwiftTensor(); 
    }
    SwiftTensor operator-(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator*(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator/(const SwiftTensor& t)const { return SwiftTensor(); }
};

#endif
