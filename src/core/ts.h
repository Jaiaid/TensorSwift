#ifndef _TS_H
#define _TS_H

#ifdef BUILD_OPENMP
#define SYS_PARAM_CPUCOUNT 8
#include <omp.h>
#endif

#include <core/storage.h>

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

        // private method to dot product of two floating point array
        float vecprod(float* bf1, float* bf2, int rowsize)const;

        // private method to multiply with another tensor
        SwiftTensor multiply(const SwiftTensor& t)const;

    public:
        std::vector<int> shape;
        
        SwiftTensor();
        
        SwiftTensor(const std::vector<int>& shape);

        SwiftTensor(const std::vector<float>& data, const std::vector<int>& shape);

        SwiftTensor(std::shared_ptr<Storage> storage_ptr, const std::vector<int>& new_shape);

        // [] overload to set value at particular entry
        void set(int idx, float val)const;

        // [] overload to set value at particular entry
        void set(const std::vector<int>& idx_list, float val)const;
        
        // return total number of elements
        int size()const;
        
        // get the storage buffer to read
        const Storage& get_storage()const;

        // to get device
        // currently there is no way to provide device type when constructing tensor
        // TODO
        // create constructor to provide device type at instantiation
        STORAGE_DEVICE get_device();

        // return a new instance with changed view but with same storage
        SwiftTensor view(const std::vector<int>& shape);

        // return a transposed view of the tensor, with same storage
        SwiftTensor get_T()const;

        // get stride at different dimension
        const std::vector<int>& get_stride_list()const;

        // [] overload to get value at particular entry
        float operator[](int idx)const;

        // [] overload to get value at particular entry
        float operator[](const std::vector<int>& idx_list)const;

        SwiftTensor sum ()const;

        // element wise addition, subtraction, multiplication and division
        // considers the underlying buffer as flattened array and  corresponding element
        // works only for tesnor with same size (shape may be different)
        // if shape is different error will be thrown
        SwiftTensor operator+(const SwiftTensor& t)const;

        SwiftTensor operator-(const SwiftTensor& t)const;

        SwiftTensor operator*(const SwiftTensor& t)const;

        SwiftTensor operator/(const SwiftTensor& t)const;

        // matrix multiplication of two tensor
        SwiftTensor matmul (const SwiftTensor& t)const;

        // matrix multiplication can be thought as multiple dot product
        // we kept it as public as many time 
        // the operation of matrix multiplication may be said as dot product due to parameter shape
        SwiftTensor dot (const SwiftTensor& t)const;

        // element wise add the floating number
        SwiftTensor operator+(const float num)const;

        SwiftTensor operator-(const float num)const;

        SwiftTensor operator*(const float num)const;

        SwiftTensor operator/(const float num)const;

        // element wise add, sub, mul or div the floating number
        // floating point number will be lhs parameter
        // declared as friend to ease access of buffer
        friend SwiftTensor operator+(const float num, const SwiftTensor& t);

        friend SwiftTensor operator-(const float num, const SwiftTensor& t);

        friend SwiftTensor operator*(const float num, const SwiftTensor& t);

        friend SwiftTensor operator*(const float num, const SwiftTensor& t);

        friend SwiftTensor operator/(const float num, const SwiftTensor& t);
    };
    // redeclared the friend function here again to stop compiler warning that "... has not been declared within ‘ts’"
    SwiftTensor operator+(const float num, const SwiftTensor& t);

    SwiftTensor operator-(const float num, const SwiftTensor& t);

    SwiftTensor operator*(const float num, const SwiftTensor& t);

    SwiftTensor operator/(const float num, const SwiftTensor& t);

    // function to convert a tensor to str
    // will be helpful to print the tensor data
    std::string to_str(const SwiftTensor& d);
}
#endif
