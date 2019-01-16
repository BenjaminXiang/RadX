#pragma once 
#include "radx_core.hpp"

// TODO: 
// - getting features and properties
// - getting vendor naming
// - detecting what is GPU

namespace radx {

    enum Vendor {
        UNIVERSAL = 0,
        AMD,
        NVIDIA,
        INTEL,

        RX_VEGA,
        NV_TURING,
    };

    inline auto getVendorName( const uint32_t& vendorID ) {
        auto shaderDir = radx::Vendor::UNIVERSAL;
        switch (vendorID) {
        case 4318:
            shaderDir = radx::Vendor::NVIDIA;
            break;
        case 4098:
            shaderDir = radx::Vendor::AMD;
            break;
        case 32902:
            shaderDir = radx::Vendor::INTEL;
            break;
        };
        return shaderDir;
    };

    class PhysicalDeviceHelper : public std::enable_shared_from_this<PhysicalDeviceHelper> {
    protected:
        vk::PhysicalDevice physicalDevice = {};
        vk::PhysicalDeviceFeatures2 features = {};
        vk::PhysicalDeviceProperties2 properties = {};
        std::vector<uint32_t> queueFamilyIndices = {};
        VmaAllocator allocator = {};

        // required (if there is no, will generated)
        radx::Vendor vendor = radx::Vendor::NV_TURING;
        
        virtual VkResult getFeaturesWithProperties(){
            this->features = physicalDevice.getFeatures2();
            this->properties = physicalDevice.getProperties2();
            return VK_SUCCESS;
        };

        virtual VkResult getVendorName(){
            this->vendor = radx::getVendorName(this->properties.properties.vendorID);
            if (this->features.features.shaderInt16) {
                if (this->vendor == radx::Vendor::AMD) this->vendor = radx::Vendor::RX_VEGA;
                if (this->vendor == radx::Vendor::NVIDIA) this->vendor = radx::Vendor::NV_TURING;
            };
            return VK_SUCCESS;
        };

    public:
        friend radx::Device;

        // require to generate both VMA and vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice) : physicalDevice(physicalDevice) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties(), this->getVendorName();
        };

        // require vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice, const VmaAllocator& allocator) : physicalDevice(physicalDevice), allocator(allocator) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties(), this->getVendorName();
            this->allocator = allocator;
        };

        // don't need to do anything 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice, const VmaAllocator& allocator, const radx::Vendor& vendor) : physicalDevice(physicalDevice), vendor(vendor), allocator(allocator) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties();
            this->allocator = allocator;
            this->vendor = vendor;
        };

        // getter of vendor name 
        operator radx::Vendor&() { return vendor; };
        operator const radx::Vendor&() const { return vendor; };

        // vk::PhysicalDevice caster
        operator vk::PhysicalDevice&() { return physicalDevice; };
        operator const vk::PhysicalDevice&() const { return physicalDevice; };
    };

    class Device : public std::enable_shared_from_this<Device> {
    protected:
        vk::Device device;
        
        // descriptor set layout 
        vk::DescriptorPool descriptorPool = {}; vk::PipelineCache pipelineCache = {};
        //vk::DescriptorSetLayout sortInputLayout, sortInterfaceLayout;

        std::vector<vk::DescriptorSetLayout> descriptorLayouts = {};
        uint32_t sortInput = 1, sortInterface = 0;

        std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper;
        VmaAllocator allocator = {};

    public:
        const std::vector<vk::DescriptorSetLayout>& getDescriptorSetLayoutSupport() const {
            return descriptorLayouts;
        };

        std::shared_ptr<Device> setDescriptorPool(vk::DescriptorPool& descriptorPool){
            this->descriptorPool = descriptorPool;
            return shared_from_this();
        };

        std::shared_ptr<Device> initialize(vk::Device& device, std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper){
            this->physicalHelper = physicalHelper;
            this->device = device;

            // get VMA allocator for device
            if (this->physicalHelper->allocator) 
            {
                this->allocator = this->physicalHelper->allocator;
            };

            
            if (!this->allocator) 
            {
                // load API calls for context
                volkLoadDevice(vk::Device(*this));

                // create VMA memory allocator (with Volk support)
#ifdef VOLK_H_
                VolkDeviceTable vktable;
                volkLoadDeviceTable(&vktable, vk::Device(*this));

                // VMA functions with Volk compatibility
                VmaVulkanFunctions vfuncs = {};
                vfuncs.vkAllocateMemory = vktable.vkAllocateMemory;
                vfuncs.vkBindBufferMemory = vktable.vkBindBufferMemory;
                vfuncs.vkBindImageMemory = vktable.vkBindImageMemory;
                vfuncs.vkCreateBuffer = vktable.vkCreateBuffer;
                vfuncs.vkCreateImage = vktable.vkCreateImage;
                vfuncs.vkDestroyBuffer = vktable.vkDestroyBuffer;
                vfuncs.vkDestroyImage = vktable.vkDestroyImage;
                vfuncs.vkFreeMemory = vktable.vkFreeMemory;
                vfuncs.vkGetBufferMemoryRequirements = vktable.vkGetBufferMemoryRequirements;
                vfuncs.vkGetBufferMemoryRequirements2KHR = vktable.vkGetBufferMemoryRequirements2KHR;
                vfuncs.vkGetImageMemoryRequirements = vktable.vkGetImageMemoryRequirements;
                vfuncs.vkGetImageMemoryRequirements2KHR = vktable.vkGetImageMemoryRequirements2KHR;
                vfuncs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
                vfuncs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
                vfuncs.vkMapMemory = vktable.vkMapMemory;
                vfuncs.vkUnmapMemory = vktable.vkUnmapMemory;
                vfuncs.vkInvalidateMappedMemoryRanges = vktable.vkInvalidateMappedMemoryRanges;
#endif
                
                // create Vma allocator
                VmaAllocatorCreateInfo allocatorInfo = {};
#ifdef VOLK_H_
                allocatorInfo.pVulkanFunctions = &vfuncs;
#endif
                allocatorInfo.physicalDevice = vk::PhysicalDevice(*this->physicalHelper);
                allocatorInfo.device = vk::Device(*this);
                allocatorInfo.preferredLargeHeapBlockSize = 16 * sizeof(uint32_t);
                allocatorInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT || VMA_ALLOCATION_CREATE_MAPPED_BIT;
                allocatorInfo.pAllocationCallbacks = nullptr;
                allocatorInfo.pHeapSizeLimit = nullptr;
                vmaCreateAllocator(&allocatorInfo, &this->allocator);
            };

            // descriptor pool
            if (!this->descriptorPool) 
            {
                // pool sizes, and create descriptor pool
                std::vector<vk::DescriptorPoolSize> psizes = { };
                psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eStorageBuffer).setDescriptorCount(128));
                psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eStorageTexelBuffer).setDescriptorCount(128));
                psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eInlineUniformBlockEXT).setDescriptorCount(128));
                psizes.push_back(vk::DescriptorPoolSize().setType(vk::DescriptorType::eUniformBuffer).setDescriptorCount(128));
                this->descriptorPool = vk::Device(*this).createDescriptorPool(vk::DescriptorPoolCreateInfo().setPPoolSizes(psizes.data()).setPoolSizeCount(psizes.size()).setMaxSets(256).setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet | vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT));
            };

            // pipeline cache 
            if (!this->pipelineCache) 
            {
                this->pipelineCache = vk::Device(*this).createPipelineCache(vk::PipelineCacheCreateInfo());
            };

            { // create descriptor layouts 
                const auto pbindings = vk::DescriptorBindingFlagBitsEXT::ePartiallyBound | vk::DescriptorBindingFlagBitsEXT::eUpdateAfterBind | vk::DescriptorBindingFlagBitsEXT::eVariableDescriptorCount | vk::DescriptorBindingFlagBitsEXT::eUpdateUnusedWhilePending;
                const auto vkfl = vk::DescriptorSetLayoutBindingFlagsCreateInfoEXT().setPBindingFlags(&pbindings);
                const auto vkpi = vk::DescriptorSetLayoutCreateInfo().setPNext(&vkfl);

                {
                    const std::vector<vk::DescriptorSetLayoutBinding> _bindings = {
                        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // keys cache
                        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // values cache
                        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // radice cache
                        vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // histogram of radices (every work group)
                        vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // prefix-sum of radices (every work group)
                        vk::DescriptorSetLayoutBinding(5, vk::DescriptorType::eInlineUniformBlockEXT, 1, vk::ShaderStageFlagBits::eCompute) // inline uniform data of algorithms
                    };
                    descriptorLayouts.push_back(device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(vkpi).setPBindings(_bindings.data()).setBindingCount(_bindings.size())));
                };

                {
                    const std::vector<vk::DescriptorSetLayoutBinding> _bindings = {
                        vk::DescriptorSetLayoutBinding(0 , vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // keys in
                        vk::DescriptorSetLayoutBinding(1 , vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute), // values in
                        vk::DescriptorSetLayoutBinding(2 , vk::DescriptorType::eInlineUniformBlockEXT, 1, vk::ShaderStageFlagBits::eCompute)
                    };
                    descriptorLayouts.push_back(device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo(vkpi).setPBindings(_bindings.data()).setBindingCount(_bindings.size())));
                };
            };

            return shared_from_this();
        };

        // queue family indices
        std::vector<uint32_t>& queueFamilyIndices() {return physicalHelper->queueFamilyIndices;};
        const std::vector<uint32_t>& queueFamilyIndices() const {return physicalHelper->queueFamilyIndices;};

        // getter of shared_ptr physical device helper
        operator std::shared_ptr<radx::PhysicalDeviceHelper>&(){ return physicalHelper; };
        operator const std::shared_ptr<radx::PhysicalDeviceHelper>&() const { return physicalHelper; };

        // get physical device helper
        std::shared_ptr<radx::PhysicalDeviceHelper>& getPhysicalHelper(){ return *this; };
        const std::shared_ptr<radx::PhysicalDeviceHelper>& getPhysicalHelper() const { return *this; };

        // vk::DescriptorPool caster
        operator vk::DescriptorPool&() { return descriptorPool; };
        operator const vk::DescriptorPool&() const { return descriptorPool; };

        // vk::PipelineCache caster
        operator vk::PipelineCache&() { return pipelineCache; };
        operator const vk::PipelineCache&() const { return pipelineCache; };

        // vk::Device caster
        operator vk::Device&() { return device; };
        operator const vk::Device&() const { return device; };

        // VmaAllocator caster
        operator VmaAllocator&() { return physicalHelper->allocator; };
        operator const VmaAllocator&() const { return physicalHelper->allocator; };
    };
};
