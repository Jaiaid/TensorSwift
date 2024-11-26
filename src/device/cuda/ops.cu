#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include <device/cuda/ops.h>


__global__ void elemwise_add(float *devptr1, float *devptr2, float *result_devptr, size_t *size_ptr)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < (int)*size_ptr) {
        result_devptr[tid] = devptr1[tid] + devptr2[tid];
    }
}

__global__ void elemwise_sub(float *devptr1, float *devptr2, float *result_devptr, size_t *size_ptr)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < (int)*size_ptr) {
        result_devptr[tid] = devptr1[tid] - devptr2[tid];
    }
}

__global__ void elemwise_mul(float *devptr1, float *devptr2, float *result_devptr, size_t *size_ptr)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < (int)*size_ptr) {
        result_devptr[tid] = devptr1[tid] * devptr2[tid];
    }
}

__global__ void elemwise_div(float *devptr1, float *devptr2, float *result_devptr, size_t *size_ptr)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < (int)*size_ptr) {
        result_devptr[tid] = devptr1[tid] / devptr2[tid];
    }
}

void cuda_elementwise_ops_wrapper(
    void *devptr1, void *devptr2, void *result_devptr, size_t size, CUDAELEMWISEOP_TYPE op_type)
{
    int blocksize = CUDA_BLOCKSIZE;
    int gridsize = (size + blocksize - 1) / blocksize;

    float *dataptr1 = (float *)((char *)devptr1 + sizeof(size_t));
    float *dataptr2 = (float *)((char *)devptr2 + sizeof(size_t));
    float *result_dataptr = (float *)((char *)result_devptr + sizeof(size_t));
    size_t *size_ptr = (size_t *) devptr1;

    if (op_type == CUDAELEMWISEOP_TYPE::ADD) {
        elemwise_add<<<gridsize, blocksize>>>(dataptr1, dataptr2, result_dataptr, size_ptr);
    }
    else if (op_type == CUDAELEMWISEOP_TYPE::SUB) {
        elemwise_sub<<<gridsize, blocksize>>>(dataptr1, dataptr2, result_dataptr, size_ptr);
    }
    else if (op_type == CUDAELEMWISEOP_TYPE::MUL) {
        elemwise_mul<<<gridsize, blocksize>>>(dataptr1, dataptr2, result_dataptr, size_ptr);
    }
    else if (op_type == CUDAELEMWISEOP_TYPE::DIV) {
        elemwise_div<<<gridsize, blocksize>>>(dataptr1, dataptr2, result_dataptr, size_ptr);
    }

    // wait for kernel to finish
    cudaDeviceSynchronize();
}
