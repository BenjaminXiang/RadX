#pragma once 
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <future>
#include <fstream>

// include vulkan API 
#include <volk/volk.h>
#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

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
