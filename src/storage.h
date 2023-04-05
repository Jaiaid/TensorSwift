#ifndef _STORAGE_H
#define _STORAGE_H

#include <iostream>

enum STORAGE_DEVICE{
    CPU,
    GPU
};

class Storage
{
public:
    float* buffer;
    uint64_t size;
    STORAGE_DEVICE devtype;

    Storage()
    {
        buffer = nullptr;
        size = 0;
        devtype = STORAGE_DEVICE::CPU;
    }

    Storage(uint64_t size, STORAGE_DEVICE devtype=STORAGE_DEVICE::CPU)
    {
        if (devtype != STORAGE_DEVICE::GPU) {
            buffer = new float[size];
        }
        else {
        }
        this->size = size;
    }

    ~Storage()
    {
        if (this->size > 0) {
            if (devtype != STORAGE_DEVICE::GPU) {
                delete[] buffer;
            }
            else {
                
            }
        }
    }
};

#endif