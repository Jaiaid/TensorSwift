#include <cuda.h>
#include <cuda_runtime.h>

#include <device/cuda/matmul.h>

struct mm_meta
{
    int r1;
    int c1;
    int c2;
};


__global__ void mm(float *dataptr1, float *dataptr2, float *result_ptr, mm_meta* mm_metadata)
{
    int total_cell = mm_metadata->r1 * mm_metadata->c2; 
    int rowstride = mm_metadata->c1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (;idx < total_cell;idx += rowstride)
    {
        result_ptr[idx] = 0;
        for (int i = 0;i < mm_metadata->c1;i++)
        {
            result_ptr[idx] += dataptr1[idx * rowstride + i] * dataptr2[idx + mm_metadata->c2 * i];
        }
    }
}

void cuda_matmul(void *devptr1, void *devptr2, void *result_ptr, int r1, int c1, int r2, int c2)
{
    mm_meta *metadata_dev_ptr;
    mm_meta mm_metadata_host;
    mm_metadata_host.r1 = r1, mm_metadata_host.c1 = c1, mm_metadata_host.c2 = c2;

    cudaMalloc(&metadata_dev_ptr, sizeof(mm_meta));
    cudaMemcpy(metadata_dev_ptr, &mm_metadata_host, sizeof(mm_meta), cudaMemcpyHostToDevice);

    // our data store for GPU device is slightly different to avoid separate call for transfer of size data
    // size (# of elements) is stored at the beginning which is of size_t type
    float *dataptr1 = (float *)((char *)devptr1 + sizeof(size_t));
    float *dataptr2 = (float *)((char *)devptr2 + sizeof(size_t));
    float *dataresult = (float *)((char *)result_ptr + sizeof(size_t));

    // one row will be given to a block
    // idea is to have one entry claculated by one block
    // if we can transpose the B of A*B, we may have better locality
    // this idea worked well for CPU at least
    int blocksize = c1;
    int gridsize = (blocksize - 1) / blocksize;
    // call kernel
    mm<<<gridsize, blocksize>>>(dataptr1, dataptr2, dataresult, metadata_dev_ptr);

    // wait until finish
    cudaDeviceSynchronize();
    cudaFree(metadata_dev_ptr);
}