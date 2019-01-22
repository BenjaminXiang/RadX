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
        //operator vk::Buffer&() { return buffer; };
        operator const vk::Buffer&() const { return buffer; };

        // Allocation
        //operator VmaAllocation&() { return allocation; };
        operator const VmaAllocation&() const { return allocation; };

        // AllocationInfo
        //operator VmaAllocationInfo&() { return allocationInfo; };
        operator const VmaAllocationInfo&() const { return allocationInfo; };
        
		// 
		operator const vk::DescriptorBufferInfo&() const { return bufInfo; };

    protected:
		void * mappedData = {};
        vk::Buffer buffer;
        VmaAllocation allocation;
        VmaAllocationInfo allocationInfo;
		VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY;
        std::shared_ptr<radx::Device> device;
		vk::DescriptorBufferInfo bufInfo = {};
    };

	// TODO: buffer copying data and zero-initializer
	template<class T>
	class Vector {
	public:
		Vector(const std::shared_ptr<VmaAllocatedBuffer>& buffer, vk::DeviceSize size = 0ull, vk::DeviceSize offset = 0u) : buffer(buffer) {
			bufInfo.buffer = *buffer;
			bufInfo.offset = offset;
			bufInfo.range = size * sizeof(T);
			//this->map();
		};

		T* map() { mapped = (T*)((uint8_t*)buffer->map() + bufInfo.offset); return mapped; };
		void unmap() { buffer->unmap(); };

		T* data() { this->map(); return mapped; };
		T* data() const { return mapped; };
		size_t size() const { return size_t(bufInfo.range / sizeof(T)); };
		const vk::DeviceSize& range() const { return bufInfo.range; };
		const T& operator [] (const uintptr_t& i) const { return mapped[i]; };
		T& operator [] (const uintptr_t& i) { return mapped[i]; };
		operator const vk::DescriptorBufferInfo&() const { return bufInfo; };
		const vk::DeviceSize& offset() const { return bufInfo.offset; };
		operator const vk::Buffer&() const { return *buffer; };
		//operator vk::Buffer&() { return *buffer; };

	protected:
		T* mapped = {};
		std::shared_ptr<VmaAllocatedBuffer> buffer = {};
		vk::DescriptorBufferInfo bufInfo = {};
	};


};
