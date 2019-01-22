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
		~Device() {
			device.waitIdle(); // wait idle before than device has been destroyed
			vmaDestroyAllocator(allocator);
		};

        const std::vector<vk::DescriptorSetLayout>& getDescriptorSetLayoutSupport() const { return descriptorLayouts; };
        std::vector<vk::DescriptorSetLayout>& getDescriptorSetLayoutSupport() { return descriptorLayouts; };

        std::shared_ptr<Device> setDescriptorPool(const vk::DescriptorPool& descriptorPool) {this->descriptorPool = descriptorPool; return shared_from_this(); };
        std::shared_ptr<Device> initialize(const vk::Device& device, std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper);

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
        operator VmaAllocator&() { return allocator; };
        operator const VmaAllocator&() const { return allocator; };
    };
};
