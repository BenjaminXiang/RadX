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
            VmaMemoryUsage vmaUsage = VMA_MEMORY_USAGE_GPU_ONLY
        ){
            // Create the buffer object without memory.
            vk::BufferCreateInfo ci{};
            ci.size = dsize;
            ci.usage = bufferUsage;
            ci.sharingMode = vk::SharingMode::eExclusive;
            ci.queueFamilyIndexCount = device->queueFamilyIndices().size();
            ci.pQueueFamilyIndices = device->queueFamilyIndices().data();

            // 
            VmaAllocationCreateInfo aci{};
            aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
            aci.usage = vmaUsage;

            //
            vmaCreateBuffer(*device, (VkBufferCreateInfo*)&ci, &aci, (VkBuffer*)&buffer, &allocation, &allocationInfo);
        };

        // Get mapped memory
        void* map(){ return allocationInfo.pMappedData; };
        void unmap(){  }; //unsupported currently

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
        vk::Buffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo allocationInfo;
        std::shared_ptr<radx::Device> device;
    };

};
