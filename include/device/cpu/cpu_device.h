#ifndef _CPU_DEVICE_H_
#define _CPU_DEVICE_H_

#include <device/device.h>

namespace cpudev
{
    class CPUDev:public device::Device
    {
        float* buffer;
    public:
        CPUDev(size_t sizebytes);

        void alloc(size_t size);
        void dealloc();

        void read(void *dst_ptr) {};
        void write(void *src_ptr) {};

        float* get_buffer()
        {
            return this->buffer;
        }
    };
}

#endif