#ifndef _CUDA_DEVICE_H_
#define _CUDA_DEVICE_H_

#include <device/device.h>

namespace cudadev
{
    class CUDADev:public device::Device
    {
        float* devbuffer_ptr;

    public:
        CUDADev(size_t sizebytes, int device_id = 0);

        int get_devid();

        void alloc(size_t sizebytes);
        void dealloc();

        void read(void *dst_ptr);
        void write(void *src_ptr);

        float* get_buffer()
        {
            return this->devbuffer_ptr;
        }
    };
}
#endif