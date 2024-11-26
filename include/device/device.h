#ifndef _DEVICE_H_
#define _DEVICE_H_

#include <iostream>


namespace device
{
    enum STORAGE_DEVICE{
        CPU,
        GPU
    };

    class Device
    {
    protected:
        int device_id;
        // size needed for transfer api call
        size_t sizebytes;
        std::string device_name;
    public:
        Device(){};
        Device(int device_id) 
        {
            this->device_id = device_id;
        }

        virtual std::string get_name()
        {
            return this->device_name;
        }

        
        /**
         * We assume if allocation is called with larger size
         * previous data will be destroyed and new data space allocated
         */
        virtual void alloc(size_t size) = 0;
        virtual void dealloc() = 0;

        virtual void read(void *dst_ptr) = 0;
        virtual void write(void *src_ptr) = 0;
        
        virtual inline float* get_buffer() = 0;
    };
}

#endif