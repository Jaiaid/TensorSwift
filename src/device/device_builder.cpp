#include <memory>

#include <device/device.h>
#include <device/device_builder.h>
#include <device/cpu/cpu_device.h>

#ifdef BUILD_CUDA
#include <device/cuda/cuda_device.h>
#endif

const std::vector<std::string> device::DeviceBuilder::devname_prefix_list = {"cpu", "cuda"};

std::unique_ptr<device::Device> device::DeviceBuilder::build_device(std::string devname, size_t sizebytes)
{
    #ifdef BUILD_CUDA
    if (devname.find("cuda") == 0) {
        int device_id = 0;
        if (devname.find(":") != std::string::npos) {
            device_id = std::stoi(devname.substr(devname.find(":") + 1));
        }
        return std::make_unique<cudadev::CUDADev>(sizebytes, device_id);
    }
    #endif
    // if nothing else create CPU device
    return std::make_unique<cpudev::CPUDev>(sizebytes);
}