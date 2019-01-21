#pragma once 

// include vulkan API 
#include "radx_device.hpp"

namespace radx {
    
    class VmaAllocatedBuffer: public std::enable_shared_from_this<VmaAllocatedBuffer>  {
    public:
        VmaAllocatedBuffer();
        VmaAllocatedBuffer(
            const std::shared_ptr<radx::Device>& device, 
            vk::DeviceSize dsize = sizeof(uint32_t), 
            vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, 
            VmaMemoryUsage vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY, bool alwaysMapped = false
        );

        // Get mapped memory
        void* map();

		// GPU unmap memory
        void unmap();

        // vk::Device caster
        operator vk::Buffer&() { return buffer; };
        operator const vk::Buffer&() const { return buffer; };

        // Allocation
        operator VmaAllocation&() { return allocation; };
        operator const VmaAllocation&() const { return allocation; };

        // AllocationInfo
        operator VmaAllocationInfo&() { return allocationInfo; };
        operator const VmaAllocationInfo&() const { return allocationInfo; };
        
    protected:
		void * mappedData = {};
        vk::Buffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo allocationInfo;
		VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY;
        std::shared_ptr<radx::Device> device;
    };

};
