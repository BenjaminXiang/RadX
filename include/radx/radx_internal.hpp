#pragma once 
#include "radx_core.hpp"
#include "radx_device.hpp"
#include "radx_shaders.hpp"

namespace radx {

    // general command buffer pipeline barrier
    static inline void commandBarrier(vk::CommandBuffer cmdBuffer) {
        vk::MemoryBarrier memoryBarrier = {};
        memoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eColorAttachmentWrite;
        memoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead  | vk::AccessFlagBits::eMemoryRead  | vk::AccessFlagBits::eTransferRead  | vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead;
        cmdBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eVertexInput,
            vk::DependencyFlagBits::eByRegion,
            {memoryBarrier},{},{});
    };



    class Interface { // used for connection between algorithms and storage
    public:
        vk::Buffer extraKeys = {}, references = {};
        vk::DescriptorSet descriptorSet = {};

        // internal 
        size_t maxElementCount = 1024*1024;
    };

    class Algorithm : public std::enable_shared_from_this<Algorithm> {
        protected:
            std::shared_ptr<radx::Device> device;

        public:
            uint32_t groupX = 1, groupY = 1;
            vk::Pipeline histogramPipeline, workloadPipeline, permutePipeline;
            vk::PipelineLayout pipelineLayout;

            virtual std::shared_ptr<Algorithm> initialize(std::shared_ptr<radx::Device>& device);

            // can be used by children 
            virtual operator Algorithm&() { return *this; };
            virtual operator const Algorithm&() const { return *this; };
    };

    template <class T>
    class Sort : public std::enable_shared_from_this<Sort<T>> {
        protected:
            std::shared_ptr<T> algorithm;
            std::shared_ptr<radx::Device> device;
            std::shared_ptr<radx::Interface> interface;
            
            // input descriptor set
            vk::DescriptorSet inputDescriptorSet;

            // input buffers and element counts
            // TODO: add dedicated descriptor set helper 
            size_t elementCount = 0;
            vk::DescriptorBufferInfo keysBufferInfo;
            vk::DescriptorBufferInfo valuesBufferInfo;

        public:
            virtual Sort<T>& initialize(std::shared_ptr<radx::Device>& device, const std::shared_ptr<T>& algorithm) {
                this->device = device, this->algorithm = algorithm;
                return *this;
            };

            virtual Sort<T>& initialize(std::shared_ptr<radx::Device>& device, const std::shared_ptr<radx::Algorithm>& algorithm) {
                this->initialize(device, std::dynamic_pointer_cast<T>(algorithm));
                return *this;
            };

            // for building arguments 
            virtual Sort<T>& setKeys(vk::DescriptorBufferInfo keys = {}){ this->keysBufferInfo = keys; return *this; };
            virtual Sort<T>& setValues(vk::DescriptorBufferInfo values = {}){ this->valuesBufferInfo = values; return *this; };
            virtual Sort<T>& setElementCount(const size_t& elementCount = 0) { this->elementCount = elementCount; };

            // 
            virtual VkResult buildCommand(vk::CommandBuffer& cmdBuf){
                std::vector<vk::DescriptorSet> descriptors = {this->interface->descriptorSet, this->inputDescriptorSet};
                cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, algorithm->pipelineLayout, 0, descriptors, {});
                
                for (uint32_t I=0;I<4;I++) { // TODO: add support variable stage length
                    cmdBuf.pushConstants(algorithm->pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0u, sizeof(uint32_t), &I);

                    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, algorithm->histogramPipeline);
                    cmdBuf.dispatch(algorithm->groupX, algorithm->groupY, 1u);
                    commandBarrier(cmdBuf);

                    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, algorithm->workloadPipeline);
                    cmdBuf.dispatch(1u, 1u, 1u);
                    commandBarrier(cmdBuf);

                    cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, algorithm->permutePipeline);
                    cmdBuf.dispatch(algorithm->groupX, algorithm->groupY, 1u);
                    commandBarrier(cmdBuf);
                }
            };

    };

    // TODO: pipeline creation and setup for device 
    class Radix : public Algorithm, public std::enable_shared_from_this<Radix> {
        protected:
        public:
            virtual std::shared_ptr<Algorithm> initialize(std::shared_ptr<radx::Device>& device) override {
                return Algorithm::shared_from_this();
            };
    };

};
