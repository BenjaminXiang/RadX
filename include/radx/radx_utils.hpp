#pragma once 
#include "radx_core.hpp"
#include <iostream>

namespace radx {

    template <typename T>
    static inline auto sgn(const T& val) { return (T(0) < val) - (val < T(0)); }

    template<class T = uint64_t>
    static inline T tiled(const T& sz, const T& gmaxtile) {
        // return (int32_t)ceil((double)sz / (double)gmaxtile);
        return sz <= 0 ? 0 : (sz / gmaxtile + sgn(sz % gmaxtile));
    }

    template <class T>
    static inline auto strided(const size_t& sizeo) { return sizeof(T) * sizeo; }

    // read binary (for SPIR-V)
    static inline auto readBinary(const std::string& filePath ) {
        std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
        std::vector<uint32_t> data = {};
        if (file.is_open()) {
            std::streampos size = file.tellg();
            data.resize(tiled(size_t(size), sizeof(uint32_t)));
            file.seekg(0, std::ios::beg);
            file.read((char *)data.data(), size);
            file.close();
        } else {
            std::cerr << "Failure to open " + filePath << std::endl;
        }
        return data;
    };

    // read source (unused)
    static inline auto readSource(const std::string& filePath, bool lineDirective = false ) {
        std::string content = "";
        std::ifstream fileStream(filePath, std::ios::in);
        if (!fileStream.is_open()) {
            std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl; return content;
        }
        std::string line = "";
        while (!fileStream.eof()) {
            std::getline(fileStream, line);
            if (lineDirective || line.find("#line") == std::string::npos) content.append(line + "\n");
        }
        fileStream.close();
        return content;
    };

    static inline auto makeShaderModuleInfo(const std::vector<uint32_t>& code) {
        auto smi = vk::ShaderModuleCreateInfo{};
        smi.pCode = (uint32_t *)code.data();
        smi.codeSize = code.size()*4;
        smi.flags = {};
        return smi;
    };

    // create shader module
    static inline auto createShaderModuleIntrusive(const vk::Device& device, const std::vector<uint32_t>& code, vk::ShaderModule& hndl) {
        return (hndl = device.createShaderModule(makeShaderModuleInfo(code)));
    };

    static inline auto createShaderModule(const vk::Device& device, const std::vector<uint32_t>& code) {
        auto sm = vk::ShaderModule{}; return createShaderModuleIntrusive(device, code, sm); return sm;
    };

    struct FixConstruction {
        vk::PipelineShaderStageCreateInfo spi = {};
        red21::VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT sgmp = {};

        operator vk::PipelineShaderStageCreateInfo& () { return spi; };
        operator const vk::PipelineShaderStageCreateInfo& () const { return spi; };
        operator red21::VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT& () { return sgmp; };
        operator const red21::VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT& () const { return sgmp; };
    };

    // create shader module
    static inline auto&& makeComputePipelineStageInfo(const vk::Device& device, const std::vector<uint32_t>& code, const char * entry = "main", const uint32_t& subgroupSize = 0u) {
        auto f = FixConstruction{};

        f.spi = vk::PipelineShaderStageCreateInfo{};
        f.spi.flags = {};
        createShaderModuleIntrusive(device, code, f.spi.module);
        f.spi.pName = entry;
        f.spi.stage = vk::ShaderStageFlagBits::eCompute;
        f.spi.pSpecializationInfo = {};

        f.sgmp = red21::VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT{};
        f.sgmp.requiredSubgroupSize = subgroupSize;
        if (subgroupSize) f.spi.pNext = &f.sgmp;

        return std::move(f);
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const vk::PipelineShaderStageCreateInfo& spi, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}, const uint32_t& subgroupSize = 0u) {
        auto cmpi = vk::ComputePipelineCreateInfo{};
        cmpi.flags = {};
        cmpi.layout = layout;
        cmpi.stage = spi;
        cmpi.basePipelineIndex = -1;
        return device.createComputePipeline(cache, cmpi);
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const std::vector<uint32_t>& code, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}, const uint32_t& subgroupSize = 0u) {
        auto f = makeComputePipelineStageInfo(device, code, "main", subgroupSize);
        if (subgroupSize) f.spi.pNext = &f.sgmp; // fix link
        return createCompute(device, f, layout, cache, subgroupSize);
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const std::string& path, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}, const uint32_t& subgroupSize = 0u) {
        return createCompute(device, readBinary(path), layout, cache, subgroupSize);
    };

    // general command buffer pipeline barrier
    static inline void commandBarrier(const vk::CommandBuffer& cmdBuffer) {
        cvk::VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = cvk::VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.pNext = nullptr;
        memoryBarrier.srcAccessMask = cvk::VK_ACCESS_SHADER_WRITE_BIT; //| cvk::VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.dstAccessMask = cvk::VK_ACCESS_SHADER_READ_BIT | cvk::VK_ACCESS_TRANSFER_READ_BIT | cvk::VK_ACCESS_UNIFORM_READ_BIT;
        cmdBuffer.pipelineBarrier(
            //vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader, {}, { memoryBarrier }, {}, {});
    };


	// create secondary command buffers for batching compute invocations
	static inline auto createCommandBuffer(cvk::VkDevice device, cvk::VkCommandPool cmdPool, bool secondary = true, bool once = true) {
        cvk::VkCommandBuffer cmdBuffer = {};

        cvk::VkCommandBufferAllocateInfo cmdi = vk::CommandBufferAllocateInfo{};
		cmdi.commandPool = cmdPool;
		cmdi.level = secondary ? cvk::VK_COMMAND_BUFFER_LEVEL_SECONDARY : cvk::VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cmdi.commandBufferCount = 1;
		vkAllocateCommandBuffers(device, &cmdi, &cmdBuffer);

        cvk::VkCommandBufferInheritanceInfo inhi = vk::CommandBufferInheritanceInfo{};
		inhi.pipelineStatistics = cvk::VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;

        cvk::VkCommandBufferBeginInfo bgi = vk::CommandBufferBeginInfo{};
		bgi.flags = {};
		bgi.flags = once ? cvk::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : cvk::VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
		bgi.pInheritanceInfo = secondary ? &inhi : nullptr;
		vkBeginCommandBuffer(cmdBuffer, &bgi);

		return cmdBuffer;
	};

	// add dispatch in command buffer (with default pipeline barrier)
	static inline cvk::VkResult cmdDispatch(cvk::VkCommandBuffer cmd, cvk::VkPipeline pipeline, uint32_t x = 1, uint32_t y = 1, uint32_t z = 1, bool barrier = true) {
		vkCmdBindPipeline(cmd, cvk::VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdDispatch(cmd, x, y, z);
		if (barrier) {
			commandBarrier(cmd); // put shader barrier
		}
		return cvk::VK_SUCCESS;
	};

	// low level copy command between (prefer for host and device)
	static inline cvk::VkResult cmdCopyBufferL(cvk::VkCommandBuffer cmd, cvk::VkBuffer srcBuffer, cvk::VkBuffer dstBuffer, const std::vector<vk::BufferCopy>& regions, std::function<void(cvk::VkCommandBuffer)> barrierFn = commandBarrier) {
		if (srcBuffer && dstBuffer && regions.size() > 0) {
			vk::CommandBuffer(cmd).copyBuffer(srcBuffer, dstBuffer, regions); barrierFn(cmd); // put copy barrier
		};
		return cvk::VK_SUCCESS;
	};


	// short data set with command buffer (alike push constant)
	template<class T>
	static inline cvk::VkResult cmdUpdateBuffer(cvk::VkCommandBuffer cmd, cvk::VkBuffer dstBuffer, cvk::VkDeviceSize offset, const std::vector<T>& data) {
		vk::CommandBuffer(cmd).updateBuffer(dstBuffer, offset, data);
		//updateCommandBarrier(cmd);
		return cvk::VK_SUCCESS;
	};

	// short data set with command buffer (alike push constant)
	template<class T>
	static inline cvk::VkResult cmdUpdateBuffer(cvk::VkCommandBuffer cmd, cvk::VkBuffer dstBuffer, cvk::VkDeviceSize offset, cvk::VkDeviceSize size, const T* data) {
		vk::CommandBuffer(cmd).updateBuffer(dstBuffer, offset, size, data);
		//updateCommandBarrier(cmd);
		return cvk::VK_SUCCESS;
	};


	// template function for fill buffer by constant value
	// use for create repeat variant
	template<uint32_t Rv>
	static inline cvk::VkResult cmdFillBuffer(cvk::VkCommandBuffer cmd, cvk::VkBuffer dstBuffer, cvk::VkDeviceSize size = VK_WHOLE_SIZE, cvk::VkDeviceSize offset = 0) {
		vk::CommandBuffer(cmd).fillBuffer(vk::Buffer(dstBuffer), offset, size, Rv);
		//updateCommandBarrier(cmd);
		return cvk::VK_SUCCESS;
	};


	// submit command (with async wait)
	static inline void submitCmd(cvk::VkDevice device, cvk::VkQueue queue, std::vector<cvk::VkCommandBuffer> cmds, vk::SubmitInfo smbi = {}) {
		// no commands 
		if (cmds.size() <= 0) return;

		smbi.commandBufferCount = cmds.size();
		smbi.pCommandBuffers = (vk::CommandBuffer*)cmds.data();

        cvk::VkFence fence = {}; cvk::VkFenceCreateInfo fin = vk::FenceCreateInfo{};
        cvk::vkCreateFence(device, &fin, nullptr, &fence);
        cvk::vkQueueSubmit(queue, 1, (const cvk::VkSubmitInfo*)& smbi, fence);
        cvk::vkWaitForFences(device, 1, &fence, true, INT64_MAX);
        cvk::vkDestroyFence(device, fence, nullptr);
	};

	// once submit command buffer
	static inline void submitOnce(cvk::VkDevice device, cvk::VkQueue queue, cvk::VkCommandPool cmdPool, std::function<void(cvk::VkCommandBuffer)> cmdFn = {}, vk::SubmitInfo smbi = {}) {
		auto cmdBuf = createCommandBuffer(device, cmdPool, false); cmdFn(cmdBuf); cvk::vkEndCommandBuffer(cmdBuf);
		submitCmd(device, queue, { cmdBuf }); vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuf); // free that command buffer
	};

	// submit command (with async wait)
	static inline void submitCmdAsync(cvk::VkDevice device, cvk::VkQueue queue, std::vector<cvk::VkCommandBuffer> cmds, std::function<void()> asyncCallback = {}, vk::SubmitInfo smbi = {}) {
		// no commands 
		if (cmds.size() <= 0) return;

		smbi.commandBufferCount = cmds.size();
		smbi.pCommandBuffers = (const vk::CommandBuffer*)cmds.data();

        cvk::VkFence fence = {}; cvk::VkFenceCreateInfo fin = vk::FenceCreateInfo{};
        cvk::vkCreateFence(device, &fin, nullptr, &fence);
        cvk::vkQueueSubmit(queue, 1, (const cvk::VkSubmitInfo*)& smbi, fence);
        cvk::vkWaitForFences(device, 1, &fence, true, INT64_MAX);
        cvk::vkDestroyFence(device, fence, nullptr);
		if (asyncCallback) asyncCallback();
	};

	// once submit command buffer
	static inline void submitOnceAsync(cvk::VkDevice device, cvk::VkQueue queue, cvk::VkCommandPool cmdPool, std::function<void(cvk::VkCommandBuffer)> cmdFn = {}, std::function<void(cvk::VkCommandBuffer)> asyncCallback = {}, vk::SubmitInfo smbi = {}) {
		auto cmdBuf = createCommandBuffer(device, cmdPool, false); cmdFn(cmdBuf); cvk::vkEndCommandBuffer(cmdBuf);
		submitCmdAsync(device, queue, { cmdBuf }, [&]() {
			asyncCallback(cmdBuf); // call async callback
            cvk::vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuf); // free that command buffer
		});
	};

	template <class T> static inline auto makeVector(const T * ptr, size_t size = 1) { std::vector<T>v(size); memcpy(v.data(), ptr, strided<T>(size)); return v; };

	// create fence function
	static inline vk::Fence createFence(cvk::VkDevice device, bool signaled = true) {
		vk::FenceCreateInfo info = {};
		if (signaled) info.setFlags(vk::FenceCreateFlagBits::eSignaled);
		return vk::Device(device).createFence(info);
	};



};
