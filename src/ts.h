#ifndef _TS_H
#define _TS_H

#include <memory>
#include <iostream>

#include "storage.h"

class SwiftTensor
{
    std::shared_ptr<Storage> storage_ptr;
    // dim_offset will contain the stride between same dimension level
    // e.g. for a 2D array of [5][6]
    // each entry will have 1 element gap between two 2nd dimension
    // [0][4] and [0][5]
    // [1][4] and [0][4] has 6 element gap
    std::vector<int> dim_offset;

    // should call after each event of shape reset
    void recalc_dim()
    {
        this->dim_offset = std::vector<int>(this->shape.size());
        // for last axis stride is always 1 element
        this->dim_offset[this->dim_offset.size()-1] = 1;
        int stride_accum = 1;
        // do from reverse
        for (int i=this->dim_offset.size()-2;i>=0;i--)
        {
            stride_accum *= this->shape[i+1];
            this->dim_offset[i] = stride_accum;
        }
    }
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
        this->recalc_dim();
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
            this->recalc_dim();
        }
    }
    
    // return a new instance with changed view but with same storage
    SwiftTensor view(const std::vector<int>& shape)
    {
        SwiftTensor new_view = SwiftTensor(this->storage_ptr, shape);
        return new_view;
    }

    // return total number of element
    int size()const
    {
        return this->storage_ptr->size;
    }

    // get the storage buffer to read
    const Storage& get_storage()const
    {
        return *(this->storage_ptr);
    }

    // get stride at different dimension
    const std::vector<int>& get_stride_list()const
    {
        return this->dim_offset;
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
        // t.size() == this->size()
        // generate a tensor with independent storage but same shape
        SwiftTensor result = SwiftTensor(this->shape);

        // addition loop
        float* buffer1 = this->storage_ptr->buffer;
        float* buffer2 = t.get_storage().buffer;
        for (int i=0;i<this->size();i++) 
        {
            result.get_storage().buffer[i] = buffer1[i] + buffer2[i];
        }

        return result;
    }

    SwiftTensor operator-(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator*(const SwiftTensor& t)const { return SwiftTensor(); }
    SwiftTensor operator/(const SwiftTensor& t)const { return SwiftTensor(); }

    // [] overload to get value at particular entry
    float operator[](int idx)const
    {
        if (idx >= this->size()) {
            return this->storage_ptr->buffer[0];
        }
        return this->storage_ptr->buffer[idx];
    }

    // [] overload to get value at particular entry
    float operator[](const std::vector<int>& idx_list)const
    {
        int offset = 0;
        // TODO?
        // throw error?
        if (idx_list.size() > this->dim_offset.size()) {
            return this->storage_ptr->buffer[0];
        }

        for(uint64_t i=0;i<this->dim_offset.size();i++)
        {
            offset += this->dim_offset[i]*idx_list[i];
        }
        // check if valid offset
        // TODO?
        // throw error?
        if (offset >= this->size()) {
            return this->storage_ptr->buffer[0];
        }

        return this->storage_ptr->buffer[offset];
    }

    // [] overload to get value at particular entry
    void set(int idx, float val)const
    {
        if (idx < this->size()) {
            this->storage_ptr->buffer[idx] = val;
        }
        
    }

    // [] overload to get value at particular entry
    void set(const std::vector<int>& idx_list, float val)const
    {
        int offset = 0;
        // TODO?
        // throw error?
        if (idx_list.size() == this->dim_offset.size()) {
            for(uint64_t i=0;i<this->dim_offset.size();i++)
            {
                offset += this->dim_offset[i]*idx_list[i];
            }
            // check if valid offset
            // TODO?
            // throw error?
            if (offset < this->size()) {
                this->storage_ptr->buffer[offset] = val;
            }
        }   
    }
};

// does not work well
// probably writing recursive function will be easier
std::string tensorswift_stringify(const SwiftTensor& d)
{
    std::string str_format;
    const float* buffer = d.get_storage().buffer;
    const std::vector<int> stride_list = d.get_stride_list();
    std::vector<char> operator_flip_list(stride_list.size(), '['); 
    int dimension_printing = 0;

    for(int i=0;i<d.size();i++)
    {
        while (dimension_printing < stride_list.size() && i%stride_list[dimension_printing] == 0)
        {
            str_format += operator_flip_list[dimension_printing];
            dimension_printing++;
        }
        dimension_printing--;
        str_format += std::to_string(buffer[i]);
    }
    return str_format;
}

#endif
