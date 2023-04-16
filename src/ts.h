#ifndef _TS_H
#define _TS_H

#ifdef BUILD_OPENMP
#define SYS_PARAM_CPUCOUNT 8
#include <omp.h>
#endif

#include "storage.h"

namespace ts
{
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
        void recalc_dim();
    public:
        std::vector<int> shape;
        
        SwiftTensor();
        
        SwiftTensor(const std::vector<int>& shape);

        SwiftTensor(const std::vector<float>& data, const std::vector<int>& shape);

        SwiftTensor(std::shared_ptr<Storage> storage_ptr, const std::vector<int>& new_shape);
        
        // return a new instance with changed view but with same storage
        SwiftTensor view(const std::vector<int>& shape);

        // return total number of element
        int size()const;

        // get the storage buffer to read
        const Storage& get_storage()const;

        // get stride at different dimension
        const std::vector<int>& get_stride_list()const;
        // to get device
        // currently there is no way to provide device type when constructing tensor
        // TODO
        // create constructor to provide device type at instantiation
        STORAGE_DEVICE get_device();

        // [] overload to get value at particular entry
        float operator[](int idx)const;

        // [] overload to get value at particular entry
        float operator[](const std::vector<int>& idx_list)const;

        // element wise addition
        // considers the underlying buffer as flattened array and add corresponding element
        // works only for tesnor with same size (shape may be different)
        SwiftTensor operator+(const SwiftTensor& t)const;

        SwiftTensor operator-(const SwiftTensor& t)const;

        SwiftTensor operator*(const SwiftTensor& t)const;

        SwiftTensor multiply(const SwiftTensor& t)const;

        float vecprod(float* bf1, float* bf2, int rowsize)const;

        SwiftTensor dot (const SwiftTensor& t)const;

        SwiftTensor matmul (const SwiftTensor& t)const;

        SwiftTensor sum (const SwiftTensor& t);

        SwiftTensor operator/(const SwiftTensor& t)const;

        // element wise add the floating number
        SwiftTensor operator+(const float num)const;

        SwiftTensor operator-(const float num)const;
        

        SwiftTensor operator*(const float num)const;

        SwiftTensor operator/(const float num)const;

        // element wise add the floating number
        // declared as friend to ease access of buffer
        friend SwiftTensor operator+(const float num, const SwiftTensor& t);

        friend SwiftTensor operator-(const float num, const SwiftTensor& t);

        friend SwiftTensor operator*(const float num, const SwiftTensor& t);

        // There is no implementation of a division of a number by SwiftTensor

        // [] overload to get value at particular entry
        void set(int idx, float val)const;

        // [] overload to get value at particular entry
        void set(const std::vector<int>& idx_list, float val)const;
    };
    // redeclared the friend function here again to stop compiler warning that "... has not been declared within ‘ts’"
    SwiftTensor operator+(const float num, const SwiftTensor& t);
    SwiftTensor operator-(const float num, const SwiftTensor& t);
    SwiftTensor operator*(const float num, const SwiftTensor& t);

    // c++ side implementation for python side __repr__ method of object
    void recursive_tensor_str_format_generation(std::string& tensor_str, std::vector<int>& stride_list, const float* buffer, int len, int& cur_idx, int cur_dimension);

    std::string tensorswift_stringify(const SwiftTensor& d);
}
#endif
