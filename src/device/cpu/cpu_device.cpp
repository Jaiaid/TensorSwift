#include <immintrin.h>

#include <device/cpu/cpu_device.h>

cpudev::CPUDev::CPUDev(size_t sizebytes)
{
    this->device_id = -1;
    this->device_name = "cpu";
    this->sizebytes = sizebytes;
    this->buffer = new float[this->sizebytes/sizeof(float)];
}

void cpudev::CPUDev::alloc(size_t sizebytes)
{
    if (this->sizebytes < sizebytes) {
        delete[] this->buffer;
        this->buffer = new float[sizebytes/sizeof(float)];
        this->sizebytes = sizebytes;
    }
}

void cpudev::CPUDev::dealloc()
{
    if (this->sizebytes > 0) {
        delete[] this->buffer;
    }
}