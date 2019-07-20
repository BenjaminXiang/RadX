#pragma once 

// include vulkan API 
//#include <intrin.h>
//#include <vulkan/vulkan.h>
//#include <volk/volk.h>
#include <vulkan/vulkan.hpp>
#include <botanic/red21.h>
//#include <volk/volk.h>
//namespace cvk {
    #include <vma/vk_mem_alloc.h>
//};
#include <fstream>

namespace radx {

    class Device;
    class PhysicalDeviceHelper;

    template<class T>
    class Sort;

    // radix sort algorithm
    class Algorithm;
    class Radix;
    class VmaAllocatedBuffer;

    // 
    class InputInterface;
    class InternalInterface;

    // radix sort templated
    using RadixSort = Sort<Radix>;

};
