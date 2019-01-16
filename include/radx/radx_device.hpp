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
        NTEL,

        RX_VEGA,
        NV_TURING,
    };

    class PhysicalDeviceHelper : public std::enable_shared_from_this<PhysicalDeviceHelper> {
        protected:
        vk::PhysicalDevice physicalDevice = {};
        vk::PhysicalDeviceFeatures2 features = {};
        vk::PhysicalDeviceProperties2 properties = {};
        std::vector<uint32_t> queueFamilyIndices = {};

        // required (if there is no, will generated)
        radx::Vendor vendor = radx::Vendor::NV_TURING;
        VmaAllocator allocator = {};
        public:
        friend radx::Device;

        virtual VkResult getFeaturesWithProperties(){
            this->features = physicalDevice.getFeatures2();
            this->properties = physicalDevice.getProperties2();
        };

        // require to generate both VMA and vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice) : physicalDevice(physicalDevice) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties();
        };

        // require vendor name 
        PhysicalDeviceHelper(const vk::PhysicalDevice& physicalDevice, const VmaAllocator& allocator) : physicalDevice(physicalDevice), allocator(allocator) {
            this->physicalDevice = physicalDevice, this->getFeaturesWithProperties();
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
            vk::DescriptorPool descriptorPool = {};
            vk::DescriptorSetLayout sortInputLayout, sortInterfaceLayout;
            std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper;

        public:
            

            std::vector<vk::DescriptorSetLayout> getDescriptorSetLayoutSupport() const {
                return {sortInterfaceLayout, sortInputLayout};
            };

            std::shared_ptr<Device> setDescriptorPool(vk::DescriptorPool& descriptorPool){
                this->descriptorPool = descriptorPool;
                return shared_from_this();
            };

            std::shared_ptr<Device> initialize(vk::Device& device, std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper){
                this->physicalHelper = physicalHelper;
                this->device = device;
                return shared_from_this();
            };

            // Queue Family indices
            std::vector<uint32_t>& queueFamilyIndices() {return physicalHelper->queueFamilyIndices;};
            const std::vector<uint32_t>& queueFamilyIndices() const {return physicalHelper->queueFamilyIndices;};

            // getter of shared_ptr physical device helper
            operator std::shared_ptr<radx::PhysicalDeviceHelper>&(){ return physicalHelper; };
            operator const std::shared_ptr<radx::PhysicalDeviceHelper>&() const { return physicalHelper; };

            std::shared_ptr<radx::PhysicalDeviceHelper>& getPhysicalHelper(){ return *this; };
            const std::shared_ptr<radx::PhysicalDeviceHelper>& getPhysicalHelper() const { return *this; };

            // vk::Device caster
            operator vk::Device&() { return device; };
            operator const vk::Device&() const { return device; };

            // Allocator
            operator VmaAllocator&() { return physicalHelper->allocator; };
            operator const VmaAllocator&() const { return physicalHelper->allocator; };
    };
};
