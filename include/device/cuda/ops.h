#ifndef _CUDA_OPS_H_
#define _CUDA_OPS_H_

#ifndef CUDA_BLOCKSIZE
#define CUDA_BLOCKSIZE 256
#endif

enum CUDAELEMWISEOP_TYPE{
    ADD,
    SUB,
    MUL,
    DIV
};

void cuda_elementwise_ops_wrapper(void *devptr1, void *devptr2, void *result_devptr, size_t size, CUDAELEMWISEOP_TYPE op_type);

#endif