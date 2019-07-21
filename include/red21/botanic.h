#pragma once 

// There is Vulkan API some extensions early access
// We Awiting For https://developer.nvidia.com/vulkan-driver Update

#include "../vulkan/vulkan.h"

namespace red21 {

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

    typedef struct VkPhysicalDeviceSubgroupSizeControlPropertiesEXT {
        cvk::VkStructureType       sType;
        void*                 pNext;
        uint32_t              minSubgroupSize;
        uint32_t              maxSubgroupSize;
        uint32_t              maxComputeWorkgroupSubgroups;
        cvk::VkShaderStageFlags    requiredSubgroupSizeStages;
    } VkPhysicalDeviceSubgroupSizeControlPropertiesEXT;

    typedef struct VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT {
        cvk::VkStructureType    sType;
        void*              pNext;
        uint32_t           requiredSubgroupSize;
    } VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT;

};
