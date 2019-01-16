#pragma once 
#include "radx_core.hpp"
#include "radx_device.hpp"
#include "radx_shaders.hpp"
#include "radx_buffer.hpp"

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
        return device.createShaderModule(makeShaderModuleInfo(code));
    };

    static inline auto createShaderModule(const vk::Device& device, const std::vector<uint32_t>& code) {
        auto sm = vk::ShaderModule{}; return createShaderModuleIntrusive(device, code, sm); return sm;
    };

    // create shader module
    static inline auto makeComputePipelineStageInfo(const vk::Device& device, const std::vector<uint32_t>& code, const char * entry = "main") {
        auto spi = vk::PipelineShaderStageCreateInfo{};
        spi.flags = {};
        createShaderModuleIntrusive(device, code, spi.module);
        spi.pName = entry;
        spi.stage = vk::ShaderStageFlagBits::eCompute;
        spi.pSpecializationInfo = {};
        return spi;
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const vk::PipelineShaderStageCreateInfo& spi, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}) {
        auto cmpi = vk::ComputePipelineCreateInfo{};
        cmpi.flags = {};
        cmpi.layout = layout;
        cmpi.stage = spi;
        cmpi.basePipelineIndex = -1;
        return device.createComputePipeline(cache, cmpi);
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const std::vector<uint32_t>& code, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}) {
        return createCompute(device, makeComputePipelineStageInfo(device, code), layout, cache);
    };

    // create compute pipelines
    static inline auto createCompute(const vk::Device& device, const std::string& path, const vk::PipelineLayout& layout, const vk::PipelineCache& cache = {}) {
        return createCompute(device, readBinary(path), layout, cache);
    };

    // general command buffer pipeline barrier
    static inline void commandBarrier(const vk::CommandBuffer& cmdBuffer) {
        vk::MemoryBarrier memoryBarrier = {};
        memoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eColorAttachmentWrite;
        memoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead  | vk::AccessFlagBits::eMemoryRead  | vk::AccessFlagBits::eTransferRead  | vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead;
        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eVertexInput,
            vk::DependencyFlagBits::eByRegion,
            {memoryBarrier},{},{});
    };



    class InternalInterface { // used for connection between algorithms and storage
    protected:
        std::shared_ptr<radx::Device> device = {};
    public:
        friend Algorithm;
        InternalInterface(){};
        InternalInterface(const std::shared_ptr<radx::Device>& device): device(device) {
            
        };

        size_t maxElementCount = 1024*1024;
        std::unique_ptr<VmaAllocatedBuffer> bufferMemory = {}; // allocated personally, once
        vk::DescriptorBufferInfo extraKeysBufferInfo, referencesBufferInfo;
        vk::DescriptorBufferInfo keysStoreBufferInfo, valuesStoreBufferInfo;
        vk::DescriptorSet descriptorSet = {};

        virtual InternalInterface& setExtraKeys(const vk::DescriptorBufferInfo& extraKeys = {}){ this->extraKeysBufferInfo = extraKeys; return *this; };
        virtual InternalInterface& setReferences(const vk::DescriptorBufferInfo& references = {}){ this->referencesBufferInfo = references; return *this; };
        virtual InternalInterface& setMaxElementCount(const size_t& elementCount = 0) { this->maxElementCount = maxElementCount; };
        
        virtual InternalInterface& buildDescriptorSet(){
            
        };
    };


    class InputInterface {
    protected:
        std::shared_ptr<radx::Device> device = {};
    public:
        friend Algorithm;
        InputInterface(){};
        InputInterface(const std::shared_ptr<radx::Device>& device): device(device) {};

        size_t elementCount = 0;
        vk::DescriptorBufferInfo keysBufferInfo, valuesBufferInfo;
        vk::DescriptorSet descriptorSet;

        // for building arguments 
        virtual InputInterface& setKeys(const vk::DescriptorBufferInfo& keys = {}){ this->keysBufferInfo = keys; return *this; };
        virtual InputInterface& setValues(const vk::DescriptorBufferInfo& values = {}){ this->valuesBufferInfo = values; return *this; };
        virtual InputInterface& setElementCount(const size_t& elementCount = 0) { this->elementCount = elementCount; };

        virtual InputInterface& buildDescriptorSet(){
            
        };
    };


    // abstract class for sorting alrgorithm
    class Algorithm : public std::enable_shared_from_this<Algorithm> {
    protected:
        std::shared_ptr<radx::Device> device;

        uint32_t groupX = 1, groupY = 1;
        std::vector<vk::Pipeline> pipelines = {};
        vk::PipelineLayout pipelineLayout;

        // internal methods (for devs)
        virtual std::shared_ptr<Algorithm> genCommand(const vk::CommandBuffer& cmdBuf, const std::unique_ptr<radx::InternalInterface>& internalInterface, const std::shared_ptr<radx::InputInterface>& inputInterface, VkResult& vkres);
        virtual std::shared_ptr<Algorithm> createInternalMemory(std::unique_ptr<radx::InternalInterface>& internalInterface, const size_t& maxElementCount = 1024 * 1024);

    public:
        friend Sort<Algorithm>;
        virtual std::shared_ptr<Algorithm> initialize(const std::shared_ptr<radx::Device>& device);

        // can be used by children 
        virtual operator Algorithm&() { return *this; };
        virtual operator const Algorithm&() const { return *this; };
    };


    template <class T>
    class Sort : public std::enable_shared_from_this<Sort<T>> {
    protected:
        std::shared_ptr<T> algorithm;
        std::shared_ptr<radx::Device> device;
        std::unique_ptr<radx::InternalInterface> internalInterface;
        //std::shared_ptr<radx::InputInterface> inputInterface;
        
    public:
        virtual Sort<T>& initialize(const std::shared_ptr<radx::Device>& device, const std::shared_ptr<T>& algorithm, const size_t& maxElementCount = 1024 * 1024) {
            this->device = device, this->algorithm = algorithm;
            this->algorithm->createInternalMemory(this->internalInterface = std::make_unique<InternalInterface>(), maxElementCount);
            return *this;
        };

        // accepts only right-based links 
        virtual Sort<T>& initialize(const std::shared_ptr<radx::Device>& device, std::shared_ptr<radx::Algorithm>&& algorithm, const size_t& maxElementCount = 1024 * 1024) {
            this->initialize(device, std::dynamic_pointer_cast<T>(algorithm), maxElementCount);
            return *this;
        };


        // TODO: add unique ptr support of input interface 
        virtual Sort<T>& genCommand(const vk::CommandBuffer& cmdBuf, std::shared_ptr<radx::InputInterface>& inputInterface){
            VkResult vkres = VK_SUCCESS; algorithm->genCommand(cmdBuf, internalInterface, inputInterface, vkres); //return vkres;
            return *this;
        };

    };

    // TODO: better vendor-based setup for device 
    class Radix : public Algorithm, public std::enable_shared_from_this<Radix> {
    protected:
        uint32_t histogram = 0, workload = 1, permute = 2, copyhack = 3, resolve = 4;

    public:
        friend Sort<Radix>;
        virtual std::shared_ptr<Algorithm> initialize(const std::shared_ptr<radx::Device>& device) override {
            std::vector<vk::DescriptorSetLayout> setLayouts = device->getDescriptorSetLayoutSupport();

            // push constant ranges
            vk::PushConstantRange pConstRange{};
            pConstRange.stageFlags = vk::ShaderStageFlagBits::eCompute;
            pConstRange.offset = 0u;
            pConstRange.size = sizeof(uint32_t)*4u;

            // pipeline layout create info
            vk::PipelineLayoutCreateInfo pplLayoutCi{};
            pplLayoutCi.setLayoutCount = setLayouts.size();
            pplLayoutCi.pSetLayouts = setLayouts.data();
            pplLayoutCi.pushConstantRangeCount = 1;
            pplLayoutCi.pPushConstantRanges = &pConstRange;

            // create pipeline layout 
            this->pipelineLayout = vk::Device(*device).createPipelineLayout(pplLayoutCi);
            this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::histogram, *device->getPhysicalHelper()), this->pipelineLayout));
            this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::workload, *device->getPhysicalHelper()), this->pipelineLayout));
            this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::permute, *device->getPhysicalHelper()), this->pipelineLayout));
            this->pipelines.push_back(createCompute(*device, radx::paths::getCorrectPath(radx::paths::copyhack, *device->getPhysicalHelper()), this->pipelineLayout));

            // return shared_ptr when needed
            return Algorithm::shared_from_this();
        };

        virtual std::shared_ptr<Algorithm> genCommand(const vk::CommandBuffer& cmdBuf, const std::unique_ptr<radx::InternalInterface>& internalInterface, const std::shared_ptr<radx::InputInterface>& inputInterface, VkResult& vkres) override {
            std::vector<vk::DescriptorSet> descriptors = {internalInterface->descriptorSet, inputInterface->descriptorSet};
            cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, this->pipelineLayout, 0, descriptors, {});
            
            for (uint32_t I=0;I<4;I++) { // TODO: add support variable stage length
                std::array<uint32_t,4> stageC = {I,0,0,0};
                cmdBuf.pushConstants(this->pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0u, sizeof(stageC), &stageC[0]);

                cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->copyhack]);
                cmdBuf.dispatch(this->groupX, 1u, 1u);
                commandBarrier(cmdBuf);

                cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->histogram]);
                cmdBuf.dispatch(this->groupX, this->groupY, 1u);
                commandBarrier(cmdBuf);

                cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->workload]);
                cmdBuf.dispatch(1u, 1u, 1u);
                commandBarrier(cmdBuf);

                cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, this->pipelines[this->permute]);
                cmdBuf.dispatch(this->groupX, this->groupY, 1u);
                commandBarrier(cmdBuf);
            }

            return Algorithm::shared_from_this();
        };

        // TODO: create sorter memory and descriptor set 
        virtual std::shared_ptr<Algorithm> createInternalMemory(std::unique_ptr<radx::InternalInterface>& internalInterface, const size_t& maxElementCount = 1024 * 1024) override {
            internalInterface->maxElementCount = maxElementCount;
            
            
            return Algorithm::shared_from_this();
        };
    };

};
