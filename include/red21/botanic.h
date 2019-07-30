#pragma once 

// There is Vulkan API some extensions early access
// We Awiting For https://developer.nvidia.com/vulkan-driver Update

#include "../vulkan/vulkan.h"

namespace red21 {

    using namespace cvk;

    #define VK_EXT_subgroup_size_control 1
    #define VK_EXT_SUBGROUP_SIZE_CONTROL_SPEC_VERSION 1
    #define VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME "VK_EXT_subgroup_size_control"

    typedef enum VkSubgroupFeatureFlagBits {
        VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001,
        VK_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002,
        VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004,
        VK_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008,
        VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010,
        VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x00000020,
        VK_SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040,
        VK_SUBGROUP_FEATURE_QUAD_BIT = 0x00000080,
        VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV = 0x00000100,
        VK_SUBGROUP_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    } VkSubgroupFeatureFlagBits;

    const cvk::VkStructureType VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT = cvk::VkStructureType(1000225000u);
    const cvk::VkStructureType VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT = cvk::VkStructureType(1000225001u);

    typedef enum VkPipelineShaderStageCreateFlagBits {
        VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT = 0x00000001,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT = 0x00000002,
        VK_PIPELINE_SHADER_STAGE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
    } VkPipelineShaderStageCreateFlagBits;
    typedef VkFlags VkPipelineShaderStageCreateFlags;

#pragma pack(push, 1)
    typedef struct VkPhysicalDeviceSubgroupSizeControlPropertiesEXT {
        cvk::VkStructureType       sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
        void*                      pNext = nullptr;
        uint32_t                   minSubgroupSize = 0u;//16u;
        uint32_t                   maxSubgroupSize = 0u;//64u;
        uint32_t                   maxComputeWorkgroupSubgroups = 0u;//36u;
        cvk::VkShaderStageFlags    requiredSubgroupSizeStages = 0u;//VK_SHADER_STAGE_COMPUTE_BIT;
    } VkPhysicalDeviceSubgroupSizeControlPropertiesEXT;

    typedef struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT {
        cvk::VkStructureType    sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        void*                   pNext = nullptr;
        uint32_t                requiredSubgroupSize = 16u;
    } VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT;
#pragma pack(pop)

};
