#pragma once 
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

// include vulkan API 
#include <radx/radx_device.hpp>

namespace vku {
    
    class VmaAllocatedBuffer {
        public:
            VmaAllocatedBuffer();
            VmaAllocatedBuffer(std::shared_ptr<radx::Device>& device){

            };
        protected:
            vk::Buffer buffer;
            VmaAllocation allocation;
            VmaAllocationInfo allocationInfo;
            std::shared_ptr<radx::Device> device;
    };

};
