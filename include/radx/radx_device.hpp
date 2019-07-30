#pragma once 
#include "radx_core.hpp"
#include "radx_shaders.hpp"

// TODO: 
// - getting features and properties
// - getting vendor naming
// - detecting what is GPU

namespace radx {

    class PhysicalDeviceHelper : public std::enable_shared_from_this<PhysicalDeviceHelper> {
    protected:
        vk::PhysicalDevice physicalDevice = {};
        vk::PhysicalDeviceFeatures2 features = {};
        vk::PhysicalDeviceProperties2 properties = {};
        std::vector<uint32_t> queueFamilyIndices = {};
        vma::VmaAllocator allocator = {};

        // required (if there is no, will generated)
        std::shared_ptr<paths::DriverWrapBase> driverWrap = {};
        
        virtual cvk::VkResult getFeaturesWithProperties(){
            this->features = physicalDevice.getFeatures2();
            this->properties = physicalDevice.getProperties2();
            return cvk::VK_SUCCESS;
        };

        virtual cvk::VkResult getVendorName(){
            driverWrap = paths::getNamedDriver(this->properties.properties.vendorID, this->features.features.shaderInt16);
            return cvk::VK_SUCCESS;
        };

    public:
        friend radx::Device;

        // require to generate both VMA and vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice) : physicalDevice(physicalDevice) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties(), this->getVendorName();
        };

        // require vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice, const vma::VmaAllocator& allocator) : physicalDevice(physicalDevice), allocator(allocator) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties(), this->getVendorName();
            this->allocator = allocator;
        };

        // getter of vendor name 
        operator const std::shared_ptr<paths::DriverWrapBase>&() const { return driverWrap; };
        std::string getPath(const std::string fpath) const { return driverWrap->getPath(fpath); };
        std::string getDriverName() const { return driverWrap->getDriverName(); };

        uint32_t getRecommendedSubgroupSize() {
            if (driverWrap->getDriverName() == "turing") { return 16u; };
            if (driverWrap->getDriverName() == "amdvlk") { return 64u; }; // but GCN have SIMD16 only
            if (driverWrap->getDriverName() == "vega10") { return 64u; }; // but GCN have SIMD16 only
            return 32u;
        };

        // vk::PhysicalDevice caster
        operator vk::PhysicalDevice&() { return physicalDevice; };
        operator const vk::PhysicalDevice&() const { return physicalDevice; };


        operator cvk::VkPhysicalDevice&() { return (cvk::VkPhysicalDevice&)physicalDevice; };
        operator const cvk::VkPhysicalDevice&() const { return (cvk::VkPhysicalDevice&)physicalDevice; };
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
        vma::VmaAllocator allocator = {};

    public:
        ~Device() {
            device.waitIdle(); // wait idle before than device has been destroyed
            vma::vmaDestroyAllocator(allocator);
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

        // 
        operator const std::shared_ptr<paths::DriverWrapBase>& () const { return *physicalHelper; };
        std::string getPath(const std::string fpath) const { return physicalHelper->getPath(fpath); };
        std::string getDriverName() const { return physicalHelper->getDriverName(); }
        uint32_t getRecommendedSubgroupSize() const { return physicalHelper->getRecommendedSubgroupSize(); };

        // vk::PhysicalDevice caster
        operator vk::PhysicalDevice& () { return *physicalHelper; };
        operator const vk::PhysicalDevice& () const { return *physicalHelper; };

        // vk::DescriptorPool caster
        operator vk::DescriptorPool&() { return descriptorPool; };
        operator const vk::DescriptorPool&() const { return descriptorPool; };

        // vk::PipelineCache caster
        operator vk::PipelineCache&() { return pipelineCache; };
        operator const vk::PipelineCache&() const { return pipelineCache; };

        // vk::Device caster
        operator vk::Device&() { return device; };
        operator const vk::Device&() const { return device; };

        // vk::Device caster
        operator cvk::VkDevice&() { return (cvk::VkDevice&)device; };
        operator const cvk::VkDevice&() const { return (cvk::VkDevice&)device; };

        // VmaAllocator caster
        operator vma::VmaAllocator&() { return allocator; };
        operator const vma::VmaAllocator&() const { return allocator; };
    };
};
