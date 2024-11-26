#ifndef _DEVICE_BUILDER_H_
#define _DEVICE_BUILDER_H_

#include <vector>
#include <string>
#include <memory>

#include <device/device.h>
#include <device/cpu/cpu_device.h>

namespace device
{
    class DeviceBuilder
    {
    public:
        static const std::vector<std::string> devname_prefix_list;

        /**
         * Return a allocated pointer to Device derived class
         * 
         */
        static std::unique_ptr<Device> build_device(std::string devname, size_t sizebytes);
        // {
        //     
        // }
    };
}

#endif