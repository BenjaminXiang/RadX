#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#define VKU_NO_GLFW
#define VMA_IMPLEMENTATION
#include "radx/radx.hpp"

namespace rad {

    class ComputeFramework {
    protected:
        vk::Queue queue = {};
        vk::Device device = {};
        vk::Instance instance = {};
        vk::PhysicalDevice physicalDevice = {};
        
        vk::Fence fence = {};
        vk::CommandPool commandPool = {};

        std::vector<vk::PhysicalDevice> physicalDevices = {};
        std::vector<uint32_t> queueFamilyIndices = {};

        // instance extensions
        std::vector<const char *> wantedExtensions = {
            "VK_KHR_get_physical_device_properties2",
            "VK_KHR_get_surface_capabilities2",
            "VK_KHR_display", "VK_KHR_surface", 
            "VK_EXT_direct_mode_display",
            "VK_EXT_swapchain_colorspace"
        };

        // default device extensions
        std::vector<const char *> wantedDeviceExtensions = {
            "VK_EXT_swapchain_colorspace",
            "VK_EXT_external_memory_host",
            "VK_EXT_sample_locations",
            "VK_EXT_conservative_rasterization",
            "VK_EXT_hdr_metadata",
            "VK_EXT_queue_family_foreign",
            "VK_EXT_sampler_filter_minmax",
            "VK_EXT_descriptor_indexing",
            "VK_EXT_scalar_block_layout",
            "VK_EXT_inline_uniform_block",

            "VK_AMD_gpu_shader_int16",
            "VK_AMD_gpu_shader_half_float",
            "VK_AMD_gcn_shader",
            "VK_AMD_buffer_marker",
            "VK_AMD_shader_info",
            "VK_AMD_texture_gather_bias_lod",
            "VK_AMD_shader_image_load_store_lod",
            "VK_AMD_shader_trinary_minmax",
            "VK_AMD_draw_indirect_count",

            "VK_KHR_16bit_storage",
            "VK_KHR_8bit_storage",
            "VK_KHR_incremental_present",
            "VK_KHR_push_descriptor",
            "VK_KHR_swapchain",
            "VK_KHR_sampler_ycbcr_conversion",
            "VK_KHR_image_format_list",
            "VK_KHR_shader_draw_parameters",
            "VK_KHR_variable_pointers",
            "VK_KHR_dedicated_allocation",
            "VK_KHR_relaxed_block_layout",
            "VK_KHR_descriptor_update_template",
            "VK_KHR_sampler_mirror_clamp_to_edge",
            "VK_KHR_storage_buffer_storage_class",
            "VK_KHR_vulkan_memory_model",
            "VK_KHR_dedicated_allocation",
            "VK_KHR_driver_properties",
            "VK_KHR_get_memory_requirements2",
            "VK_KHR_bind_memory2",
            "VK_KHR_maintenance1",
            "VK_KHR_maintenance2",
            "VK_KHR_maintenance3",
            "VK_KHX_shader_explicit_arithmetic_types",
            "VK_KHR_shader_atomic_int64",
            "VK_KHR_shader_float16_int8",
            "VK_KHR_shader_float_controls",
            
            "VK_NV_compute_shader_derivatives",
            "VK_NV_corner_sampled_image",
            "VK_NV_shader_image_footprint",
            "VK_NV_shader_subgroup_partitioned",

            "VK_NV_ray_tracing",
        };

        // instance layers
        std::vector<const char *> wantedLayers = {
            "VK_LAYER_LUNARG_assistant_layer",
            "VK_LAYER_LUNARG_standard_validation",
            "VK_LAYER_LUNARG_parameter_validation",
            "VK_LAYER_LUNARG_core_validation",

            //"VK_LAYER_LUNARG_api_dump",
            //"VK_LAYER_LUNARG_object_tracker",
            //"VK_LAYER_LUNARG_device_simulation",
            //"VK_LAYER_GOOGLE_threading",
            //"VK_LAYER_GOOGLE_unique_objects"
            //"VK_LAYER_RENDERDOC_Capture"
        };

        // default device layers
        std::vector<const char *> wantedDeviceValidationLayers = {
            "VK_LAYER_AMD_switchable_graphics"
        };

    public:
        ComputeFramework(){

        };

        vk::Instance createInstance() {

#ifdef VOLK_H_
            volkInitialize();
#endif

            auto supportedVkApiVersion = 0u;
            auto apiResult = vkEnumerateInstanceVersion(&supportedVkApiVersion);
            if (supportedVkApiVersion < VK_MAKE_VERSION(1, 1, 0)) return instance;

            // get our needed extensions
            auto installedExtensions = vk::enumerateInstanceExtensionProperties();
            auto extensions = std::vector<const char *>();
            for (auto w : wantedExtensions) {
                for (auto i : installedExtensions)
                {
                    if (std::string(i.extensionName).compare(w) == 0)
                    {
                        extensions.emplace_back(w);
                        break;
                    }
                }
            }

            // get validation layers
            auto installedLayers = vk::enumerateInstanceLayerProperties();
            auto layers = std::vector<const char *>();
            for (auto w : wantedLayers) {
                for (auto i : installedLayers)
                {
                    if (std::string(i.layerName).compare(w) == 0)
                    {
                        layers.emplace_back(w);
                        break;
                    }
                }
            }

            // app info
            auto appinfo = vk::ApplicationInfo{};
            appinfo.pNext = nullptr;
            appinfo.pApplicationName = "VKTest";
#ifndef VRT_ENABLE_VEZ_INTEROP
            appinfo.apiVersion = VK_MAKE_VERSION(1, 1, 92);
#endif

            // create instance info
            auto cinstanceinfo = vk::InstanceCreateInfo{};
            cinstanceinfo.pApplicationInfo = &appinfo;
            cinstanceinfo.enabledExtensionCount = extensions.size();
            cinstanceinfo.ppEnabledExtensionNames = extensions.data();
            cinstanceinfo.enabledLayerCount = layers.size();
            cinstanceinfo.ppEnabledLayerNames = layers.data();

            instance = vk::createInstance(cinstanceinfo);
#ifdef VOLK_H_
            volkLoadInstance(instance);
#endif

            // enumerate physical devices
            physicalDevices = instance.enumeratePhysicalDevices();
            physicalDevice = physicalDevices[0];

            // get physical device for application
            return instance;
        };


        inline const vk::PhysicalDevice& getPhysicalDevice(const uint32_t& gpuID) {
            return (physicalDevice = physicalDevices[gpuID]);
        };


        inline vk::Device createDevice(bool isComputePrior = true, std::string shaderPath = "./", bool enableAdvancedAcceleration = true) {
            // use extensions
            auto deviceExtensions = std::vector<const char *>();
            auto gpuExtensions = physicalDevice.enumerateDeviceExtensionProperties();
            for (auto w : wantedDeviceExtensions) {
                for (auto i : gpuExtensions) {
                    if (std::string(i.extensionName).compare(w) == 0) {
                        deviceExtensions.emplace_back(w); break;
                    };
                };
            };

            // use layers
            auto layers = std::vector<const char *>();
            auto deviceValidationLayers = std::vector<const char *>();
            auto gpuLayers = physicalDevice.enumerateDeviceLayerProperties();
            for (auto w : wantedLayers) {
                for (auto i : gpuLayers) {
                    if (std::string(i.layerName).compare(w) == 0) {
                        layers.emplace_back(w); break;
                    };
                };
            };

            // minimal features
            auto gStorage16 = vk::PhysicalDevice16BitStorageFeatures{};
            auto gStorage8 = vk::PhysicalDevice8BitStorageFeaturesKHR{};
            auto gDescIndexing = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT{};
            gStorage16.pNext = &gStorage8;
            gStorage8.pNext = &gDescIndexing;

            auto gFeatures = vk::PhysicalDeviceFeatures2{};
            gFeatures.pNext = &gStorage16;
            gFeatures.features.shaderInt16 = true;
            gFeatures.features.shaderInt64 = true;
            gFeatures.features.shaderUniformBufferArrayDynamicIndexing = true;
            physicalDevice.getFeatures2(&gFeatures);

            // get features and queue family properties
            //auto gpuFeatures = gpu.getFeatures();
            auto gpuQueueProps = physicalDevice.getQueueFamilyProperties();

            // queue family initial
            float priority = 1.0f;
            uint32_t computeFamilyIndex = -1, graphicsFamilyIndex = -1;
            auto queueCreateInfos = std::vector<vk::DeviceQueueCreateInfo>();

            // compute/graphics queue family
            for (auto queuefamily : gpuQueueProps) {
                computeFamilyIndex++;
                if (queuefamily.queueFlags & (vk::QueueFlagBits::eCompute)) {
                    queueCreateInfos.push_back(vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags()).setQueueFamilyIndex(computeFamilyIndex).setQueueCount(1).setPQueuePriorities(&priority));
                    queueFamilyIndices.push_back(computeFamilyIndex);
                    break;
                };
            };

            // if have supported queue family, then use this device
            if (queueCreateInfos.size() > 0) {
                // create device
                this->physicalDevice = physicalDevice;
                this->device = physicalDevice.createDevice(vk::DeviceCreateInfo().setFlags(vk::DeviceCreateFlags())
                    .setPNext(&gFeatures) //.setPEnabledFeatures(&gpuFeatures)
                    .setPQueueCreateInfos(queueCreateInfos.data()).setQueueCreateInfoCount(queueCreateInfos.size())
                    .setPpEnabledExtensionNames(deviceExtensions.data()).setEnabledExtensionCount(deviceExtensions.size())
                    .setPpEnabledLayerNames(deviceValidationLayers.data()).setEnabledLayerCount(deviceValidationLayers.size()));
            };

            // return device with queue pointer
            this->fence = this->device.createFence(vk::FenceCreateInfo().setFlags({}));
            this->commandPool = this->device.createCommandPool(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndices[0]));
            this->queue = this->device.getQueue(queueFamilyIndices[0], 0); // deferred getting of queue

            // 
            return this->device;
        };

        const vk::PhysicalDevice& getPhysicalDevice() const {return this->physicalDevice;};
        const vk::Device& getDevice() const {return this->device;};
        const vk::Queue& getQueue() const {return this->queue;};
        const vk::Fence& getFence() const {return this->fence;};
        const vk::Instance& getInstance() const {return this->instance;};
        const vk::CommandPool& getCommandPool() const {return this->commandPool;};
    };


    class TestSort : public std::enable_shared_from_this<TestSort> {
    protected:
        std::shared_ptr<radx::Device> device;
        std::shared_ptr<radx::Radix> program;
        std::shared_ptr<radx::Sort<radx::Radix>> radixSort;
        std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper;
        std::shared_ptr<radx::InputInterface> inputInterface;
        std::shared_ptr<ComputeFramework> fw;
        std::unique_ptr<radx::VmaAllocatedBuffer> vmaBuffer;
        std::unique_ptr<radx::VmaAllocatedBuffer> vmaToHostBuffer;

        // 
        const size_t elementCount = 1024;
        vk::DeviceSize keysSize = 0, valuesSize = 0;
        vk::DeviceSize keysOffset = 0, valuesOffset = 0;

    public:
        TestSort(){
            fw = std::make_shared<ComputeFramework>();
            fw->createInstance();

            // create radix sort application (RadX C++)
            physicalHelper = std::make_shared<radx::PhysicalDeviceHelper>(fw->getPhysicalDevice(0));
            device = std::make_shared<radx::Device>()->initialize(fw->createDevice(), physicalHelper);
            program = std::make_shared<radx::Radix>(), program->initialize(device);
            radixSort = std::make_shared<radx::Sort<radx::Radix>>(), radixSort->initialize(device, program);
            inputInterface = std::make_shared<radx::InputInterface>(device);
            
            { // sizes of offsets
                keysSize = elementCount * sizeof(uint32_t), valuesSize = elementCount * sizeof(uint32_t);
                keysOffset = 0, valuesOffset = keysOffset + keysSize;
            };

            // get memory size and set max element count
            vk::DeviceSize memorySize = valuesOffset + valuesSize;
            vmaBuffer = std::make_unique<radx::VmaAllocatedBuffer>(this->device, memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_TO_GPU);
            vmaToHostBuffer = std::make_unique<radx::VmaAllocatedBuffer>(this->device, memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_GPU_TO_CPU);

            // on deprecation 
            inputInterface->setElementCount(elementCount);
            inputInterface->setKeysBufferInfo(vk::DescriptorBufferInfo(*vmaBuffer, keysOffset, keysSize));
            inputInterface->setValuesBufferInfo(vk::DescriptorBufferInfo(*vmaBuffer, valuesOffset, valuesSize));
            
            // build descriptor set
            inputInterface->buildDescriptorSet();
            
            // generate random numbers and copy to buffer
            std::vector<uint32_t> randNumbers(elementCount);
            for (uint32_t i=0;i<randNumbers.size();i++) { srand(i); randNumbers[i] = rand()%0xFFFFFFFFu; };
            memcpy((uint8_t*)vmaBuffer->map()+keysOffset, randNumbers.data(), randNumbers.size()*sizeof(uint32_t)); // copy

            // command allocation 
            vk::CommandBufferAllocateInfo cci{};
            cci.commandPool = fw->getCommandPool();
            cci.commandBufferCount = 1;
            cci.level = vk::CommandBufferLevel::ePrimary;

            // generate command 
            auto cmdBuf = vk::Device(*device).allocateCommandBuffers(cci).at(0);
            cmdBuf.begin(vk::CommandBufferBeginInfo());
            radixSort->genCommand(cmdBuf, inputInterface);
            cmdBuf.copyBuffer(*vmaBuffer, *vmaToHostBuffer, { vk::BufferCopy(keysOffset, keysOffset, keysSize) }); // copy buffer to host 
            cmdBuf.end();

            // submit command 
            vk::SubmitInfo sbmi = {};
            sbmi.pCommandBuffers = &cmdBuf;
            sbmi.commandBufferCount = 1;
            auto fence = fw->getFence();
            fw->getQueue().submit(sbmi, fence);
            vk::Device(*device).waitForFences({fence}, true, INT32_MAX);
            
            // get sorted numbers
            std::vector<uint32_t> sortedNumbers(elementCount);
            memcpy(sortedNumbers.data(), (uint8_t*)vmaToHostBuffer->map()+keysOffset, sortedNumbers.size()*sizeof(uint32_t)); // copy

            // 
            std::cout << "Sorting Finished" << std::endl;
        };
    };

};
