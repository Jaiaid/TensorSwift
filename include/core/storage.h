#ifndef _STORAGE_H
#define _STORAGE_H

#include <iostream>
#include <string>
#include <memory>

#include <device/device.h>
#include <device/device_builder.h>

class Storage
{
    // data maybe brought to host or maybe send to GPU
    // it is guaranteed that a tensor will have the cpu buffer always
    // device is special case
    float* hostbuffer_ptr;
    // to indicate if latest data in device currently
    bool indev;
    // size needed for transfer api call
    size_t sizebytes;
    // for functionality if the storage also has backing from device
    std::unique_ptr<device::Device> device_ptr;
    // external to host memory backup storage device name
    std::string external_device_name;

public:
    Storage()
    {
        this->sizebytes = 0;
        this->device_ptr = nullptr;
    }

    Storage(size_t size, std::string device_name="cpu")
    {
        this->sizebytes = sizeof(float) * size;
        this->hostbuffer_ptr = new float[size];
        this->indev = false;
        this->device_ptr = nullptr;
        if (device_name != "cpu") {
            this->device_ptr = device::DeviceBuilder::build_device(device_name, this->sizebytes);
            this->external_device_name = device_name;
        }
    }

    void to(std::string dest_device_name);

    std::string get_name();

    float* get_bufferptr()
    {
        if (this->indev) {
            return this->device_ptr->get_buffer();
        }
        return this->hostbuffer_ptr;
    }

    size_t get_size() 
    {
        return this->sizebytes / sizeof(float);
    }

    ~Storage()
    {
        if (this->sizebytes > 0) {
            delete[] this->hostbuffer_ptr;
        }
        if (this->device_ptr != nullptr) {
            this->device_ptr->dealloc();
        }
    }
};

#endif