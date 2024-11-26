#include <core/storage.h>

std::string Storage::get_name()
{
    if (this->indev) {
        return this->external_device_name;
    }
    return "cpu";
}

void Storage::to(std::string dest_device_name)
{
    // data in external device, bring it back
    if (dest_device_name == "cpu" && this->indev) {
        // std::cout << "from " << this->external_device_name << " to cpu" << std::endl;
        this->device_ptr->read(hostbuffer_ptr);
        this->indev = false;
    }
    // data in cpu, needs to send it to external device (i.e., GPU)
    else if (dest_device_name == this->external_device_name && !this->indev) {
        // std::cout << "from cpu to " << this->external_device_name << std::endl;
        this->device_ptr->write(this->hostbuffer_ptr);
        this->indev = true;
    }
    // this means we need to send the data to a new device, 
    // depending on where the latest data is we may need to do extra work
    else if (dest_device_name != this->external_device_name) {
        // std::cout << "separate destination device, creating new device buffer on device " << dest_device_name << std::endl;
        std::unique_ptr<device::Device> tmp = device::DeviceBuilder::build_device(
            dest_device_name, this->sizebytes);
        // bring data to host memory if in another device
        if (this->indev) {
            // std::cout << "data in separate device moving from that" << std::endl;
            this->device_ptr->read(hostbuffer_ptr);
            // deallocate
            this->device_ptr->dealloc();
        }

        // move the ownership to class attribute
        this->device_ptr = std::move(tmp);

        // send the data to device memory
        this->device_ptr->write(this->hostbuffer_ptr);
        // set other properties
        this->indev = true;
        this->external_device_name = dest_device_name;
    }
}