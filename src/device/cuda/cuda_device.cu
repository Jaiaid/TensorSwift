#include <cuda.h>
#include <cuda_runtime.h>

#include <device/cuda/cuda_device.h>

cudadev::CUDADev::CUDADev(size_t sizebytes, int device_id):Device(device_id)
{
    cudaSetDevice(this->device_id);
    cudaMalloc(&this->devbuffer_ptr, sizebytes + sizeof(size_t));

    this->sizebytes = sizebytes;
}

int cudadev::CUDADev::get_devid()
{
    return this->device_id;
}

/**
 * cudaMalloc needs to be called only if
 * - allocated but the new size is larger than previous
 */
void cudadev::CUDADev::alloc(size_t sizebytes)
{
    if (this->sizebytes < sizebytes) {
        cudaSetDevice(this->device_id);
        cudaFree(this->devbuffer_ptr);
        cudaMalloc(&this->devbuffer_ptr, sizebytes + sizeof(size_t));
        this->sizebytes = sizebytes;
    }
}

void cudadev::CUDADev::dealloc()
{
    cudaFree(this->devbuffer_ptr);
}

void cudadev::CUDADev::read(void *dst_ptr)
{
    cudaSetDevice(this->device_id);
    cudaMemcpy(dst_ptr, this->devbuffer_ptr, this->sizebytes + sizeof(size_t), cudaMemcpyDeviceToHost);
}

void cudadev::CUDADev::write(void *src_ptr)
{
    cudaSetDevice(this->device_id);
    cudaMemcpy(this->devbuffer_ptr, src_ptr, this->sizebytes + sizeof(size_t), cudaMemcpyHostToDevice);
}