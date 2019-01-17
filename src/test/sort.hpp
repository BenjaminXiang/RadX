#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#define VKU_NO_GLFW
#include <radx/radx.hpp>

#ifdef THRUST_TESTABLE
//#define ENABLE_THRUST_BENCHMARK
#endif

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
        ComputeFramework(){};

        vk::Instance createInstance();
        vk::Device createDevice(bool isComputePrior = true, std::string shaderPath = "./", bool enableAdvancedAcceleration = true);

        const vk::PhysicalDevice& getPhysicalDevice(const uint32_t& gpuID) { return (physicalDevice = physicalDevices[gpuID]); };

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
        const size_t elementCount = 8192;
        std::vector<uint32_t> sortedNumbersThrust;

        vk::DeviceSize keysSize = 0, valuesSize = 0;
        vk::DeviceSize keysOffset = 0, valuesOffset = 0;

    public:
#ifdef ENABLE_THRUST_BENCHMARK
        void testSortingThrust();
#endif
        TestSort::TestSort();
        
    };

};
