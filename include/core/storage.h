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
        this->external_device_name = "";
        this->indev = false;
    }

    Storage(size_t size, std::string device_name="cpu")
    {
        this->sizebytes = sizeof(float) * size;
        // extra space at beginning to store the size
        // when returning the buffer pointer we will compensete for that
        // this is done to transfer the size to device within one buffer
        this->hostbuffer_ptr = new float[sizeof(size_t)/sizeof(float) + size];
        *((size_t *)this->hostbuffer_ptr) = size;

        this->indev = false;
        this->device_ptr = nullptr;
        this->external_device_name = "";
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
            // std::cout << "as data is in device, returning device ptr" << std::endl;
            return this->device_ptr->get_buffer();
        }
        return (float *)((char *)this->hostbuffer_ptr + sizeof(size_t));
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