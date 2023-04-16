#include <memory>
#include <string>
#include <vector>
// custom headers
#include "storage.h"
#include "ts.h"


void ts::SwiftTensor::recalc_dim()
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


ts::SwiftTensor::SwiftTensor(){ this->storage_ptr = std::make_shared<Storage>(); }


ts::SwiftTensor::SwiftTensor(const std::vector<int>& shape)
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


ts::SwiftTensor::SwiftTensor(const std::vector<float>& data, const std::vector<int>& shape)
{
    // calculate the size
    uint64_t intended_size = (shape.size()>0)?shape[0]:0;
    for(uint64_t i=1;i<shape.size();i++)
    {
        intended_size*=shape[i];
    }
    // TODO
    // assert if data and shape has same size 

    // create storage
    this->storage_ptr = std::make_shared<Storage>(intended_size);
    this->shape = shape;
    this->recalc_dim();

    // copy data to created storage
    float* buffer = this->storage_ptr->buffer;
    for (int i=0;i<this->size();i++) 
    {
        buffer[i] = data[i];
    }
}


ts::SwiftTensor::SwiftTensor(std::shared_ptr<Storage> storage_ptr, const std::vector<int>& new_shape)
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
ts::SwiftTensor ts::SwiftTensor::view(const std::vector<int>& shape)
{
    SwiftTensor new_view = SwiftTensor(this->storage_ptr, shape);
    return new_view;
}

// return total number of element
int ts::SwiftTensor::size()const
{
    return this->storage_ptr->size;
}


// get the storage buffer to read
const Storage& ts::SwiftTensor::get_storage()const
{
    return *(this->storage_ptr);
}

// get stride at different dimension
const std::vector<int>& ts::SwiftTensor::get_stride_list()const
{
    return this->dim_offset;
}


// to get device
// currently there is no way to provide device type when constructing tensor
// TODO
// create constructor to provide device type at instantiation
STORAGE_DEVICE ts::SwiftTensor::get_device()
{
    return this->storage_ptr->devtype;
}


// [] overload to get value at particular entry
float ts::SwiftTensor::operator[](int idx)const
{
    if (idx >= this->size()) {
        return this->storage_ptr->buffer[0];
    }
    return this->storage_ptr->buffer[idx];
}


// [] overload to get value at particular entry
float ts::SwiftTensor::operator[](const std::vector<int>& idx_list)const
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


ts::SwiftTensor ts::SwiftTensor::operator+(const SwiftTensor& t)const 
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

    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] + buffer2[i];
    }

    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator-(const SwiftTensor& t)const
{ 
    // we can add stuffs considering the buffer as 1d for + operation
    // what ever the shape, as long as size matches this should work
    // for axis specific addition, it will be done in separate function to pass axis parameter
    SwiftTensor result = SwiftTensor(this->shape);

    // subtraction loop
    float* buffer1 = this->storage_ptr->buffer;
    float* buffer2 = t.get_storage().buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] - buffer2[i];
    }

    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator*(const SwiftTensor& t)const 
{
    std::vector<int> thisshape = this->shape;
    std::vector<int> tshape = t.shape;
    
    // Considering the shape of the input tensor, the output of the product may differ
    // If any of the two tensor is a matrix, we proceed with a matrix multiplication
    if ((thisshape[1] > 1 && thisshape[2] > 1) || (tshape[1] > 1 && tshape[2] > 1))
        return this->matmul(t);

    // If all the tensors are scalar, we perform a simple multiplication
    if ((thisshape[1] == 1 && thisshape[2] == 1) && (tshape[1] == 1 && tshape[2] == 1))
        return this->multiply(t);

    // By default, we perform a dot product, element-wise multiplication, and
    // return the resulting array. If the user wants to return a single value
    // he can call the sum operation on the output array.
    return this->dot(t);
}


ts::SwiftTensor ts::SwiftTensor::multiply(const SwiftTensor& t)const 
{
    // This function only executes when both tensors have the same size
    if((this->shape[1] != t.shape[1]) || (this->shape[2] != t.shape[2])) 
    {
        throw std::invalid_argument("Imcompatible tensor dimensions.");
    }
    else {
        SwiftTensor result = SwiftTensor(this->shape);

        // multiplication loop
        float* buffer1 = this->storage_ptr->buffer;
        float* buffer2 = t.get_storage().buffer;
        #ifdef BUILD_OPENMP
        omp_set_num_threads(SYS_PARAM_CPUCOUNT);
        #pragma omp parallel for
        #endif
        for (int i=0;i<this->size();i++) 
        {
            result.get_storage().buffer[i] = buffer1[i] * buffer2[i];
        }

        return result;
    }   
}


float ts::SwiftTensor::vecprod(float* bf1, float* bf2, int rowsize)const
{
    // This vector product multiplies two arrays.
    // It is implemented because our underlying storage is as a 1D array stored row-wise.
    // Thus, to multiply the a vector by the column of a matrix, we index the matrix accordingly.
    float sum = 0;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for reduction (+:sum)
    #endif
    for (int i = 0; i < rowsize; i++)
    {
        sum = sum + bf1[i] * bf2[i];
    }
    return sum;
}


ts::SwiftTensor ts::SwiftTensor::dot (const SwiftTensor& t)const
{
    // To perform a dot poduct between two tensors, the number of column in the left tensor
    // should equal the number of rows in the right tensor.
    if(this->shape[2] != t.shape[1]) 
    {
        throw std::invalid_argument("Imcompatible tensor dimensions.");
    }

    std::vector<int> t1s = this->shape; // this tensor's shape
    std::vector<int> t2s = t.shape; // second tensor's shape
    std::vector<int> newshape = {t1s[1], t2s[2]};
    SwiftTensor result = SwiftTensor(newshape);

    float* buffer1 = this->storage_ptr->buffer;
    float* buffer2 = t.get_storage().buffer;
    int k = -1;
    int l = 0;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for(int i = 0; i < (t1s[1]*t2s[2]); i++)
    {
        if(i%t1s[2] == 0) { k++; l = 0;}
        float buffer1_slice[t1s[2]];
        float buffer2_slice[t1s[2]];
        // Take one row from the first tensor and one column from the second
        for(int j = 0; j < t1s[2]; j++)
        {
            buffer1_slice[j] = buffer1[k*t1s[2] + j];
            buffer2_slice[j] = buffer2[l+j*t1s[2]];
        }

        // Multipy the row from the first tensor with the second tensor
        result.get_storage().buffer[i] = vecprod(buffer1_slice, buffer2_slice, t1s[2]);
        l++;
    }
    return result;
}


ts::SwiftTensor ts::SwiftTensor::matmul (const SwiftTensor& t)const
{
    // As in numpy, both dot and matmul yield the same results, 
    // but the matmul is recommended for matrices.
    
    // To-Do Look into the how to make matmul work more for 2D than 1D;
    // Currently it also used the dot product as method.

    return this->dot(t);
}


ts::SwiftTensor ts::SwiftTensor::sum (const SwiftTensor& t)
{
    float sum = 0;
    std::vector<int> newshape = {1,1};
    SwiftTensor result = SwiftTensor(newshape);
    float* buffer = t.get_storage().buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for reduction (+:sum)
    #endif
    for (int i = 0; i < t.size(); i++)
    {
        sum = sum + buffer[i];
    }
    result.get_storage().buffer[0] = sum;
    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator/(const SwiftTensor& t)const
{ 
    // This function only executes when both tensors have the same size
    if((this->shape[1] != t.shape[1]) || (this->shape[2] != t.shape[2])) 
    {
        throw std::invalid_argument("Imcompatible tensor dimensions.");
    }
    else {
        SwiftTensor result = SwiftTensor(this->shape);

        // subtraction loop
        float* buffer1 = this->storage_ptr->buffer;
        float* buffer2 = t.get_storage().buffer;
        #ifdef BUILD_OPENMP
        omp_set_num_threads(SYS_PARAM_CPUCOUNT);
        #pragma omp parallel for
        #endif
        for (int i=0;i<this->size();i++) 
        {
            result.get_storage().buffer[i] = buffer1[i] / buffer2[i];
        }

        return result;
    }
}


// element wise add the floating number
ts::SwiftTensor ts::SwiftTensor::operator+(const float num)const 
{
    SwiftTensor result = SwiftTensor(this->shape);

    // addition loop
    float* buffer1 = this->storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] + num;
    }

    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator-(const float num)const
{ 
    SwiftTensor result = SwiftTensor(this->shape);

    // subtraction loop
    float* buffer1 = this->storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] - num;
    }

    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator*(const float num)const 
{
    SwiftTensor result = SwiftTensor(this->shape);

    // multiplication loop
    float* buffer1 = this->storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] * num;
    }

    return result;
}


ts::SwiftTensor ts::SwiftTensor::operator/(const float num)const 
{ 
    SwiftTensor result = SwiftTensor(this->shape);

    // division loop
    float* buffer1 = this->storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<this->size();i++) 
    {
        result.get_storage().buffer[i] = buffer1[i] / num;
    }

    return result;
}


// element wise add the floating number
ts::SwiftTensor ts::operator+(const float num, const SwiftTensor& t)
{
    SwiftTensor result = SwiftTensor(t.shape);

    // addition loop
    float* buffer1 = t.storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<t.size();i++) 
    {
        result.get_storage().buffer[i] = num + buffer1[i];
    }

    return result;
}


ts::SwiftTensor ts::operator-(const float num, const SwiftTensor& t)
{ 
    SwiftTensor result = SwiftTensor(t.shape);

    // subtraction loop
    float* buffer1 = t.storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<t.size();i++) 
    {
        result.get_storage().buffer[i] = num - buffer1[i];
    }

    return result;
}


ts::SwiftTensor ts::operator*(const float num, const SwiftTensor& t)
{ 
    SwiftTensor result = SwiftTensor(t.shape);

    // multiplication loop
    float* buffer1 = t.storage_ptr->buffer;
    #ifdef BUILD_OPENMP
    omp_set_num_threads(SYS_PARAM_CPUCOUNT);
    #pragma omp parallel for
    #endif
    for (int i=0;i<t.size();i++) 
    {
        result.get_storage().buffer[i] = num * buffer1[i];
    }

    return result;
}

// There is no implementation of a division of a number by SwiftTensor

// [] overload to get value at particular entry
void ts::SwiftTensor::set(int idx, float val)const
{
    if (idx < this->size()) {
        this->storage_ptr->buffer[idx] = val;
    }
    
}


// [] overload to get value at particular entry
void ts::SwiftTensor::set(const std::vector<int>& idx_list, float val)const
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


// this should not be this complex
// but for now it works
void ts::recursive_tensor_str_format_generation(std::string& tensor_str, std::vector<int>& stride_list, const float* buffer, int len, int& cur_idx, int cur_dimension)
{
    do{
        if (cur_dimension < (int)stride_list.size() - 1 && cur_idx%stride_list[cur_dimension]==0) {
            tensor_str += "[";
            recursive_tensor_str_format_generation(tensor_str, stride_list, buffer, len, cur_idx, cur_dimension+1);
            tensor_str += "]";
        }
        // last dimension is all number so no [...] only number
        else if (cur_dimension == (int)stride_list.size()-1) {
            tensor_str += std::to_string(buffer[cur_idx]) + ",";
            cur_idx++;
        }
    }while(cur_idx < len && (cur_dimension==0 || cur_idx%stride_list[cur_dimension-1]!=0 ));
}


std::string ts::tensorswift_stringify(const SwiftTensor& d)
{
    std::string str_format;
    const float* buffer = d.get_storage().buffer;
    std::vector<int> stride_list = d.get_stride_list();
    int idx_track = 0;

    str_format += "[";
    recursive_tensor_str_format_generation(str_format, stride_list, buffer, d.size(), idx_track, 0);
    str_format += "]";
    return str_format;
}
