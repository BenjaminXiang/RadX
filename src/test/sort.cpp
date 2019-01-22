#define VMA_IMPLEMENTATION
#define RADX_IMPLEMENTATION
#include "sort.hpp"

//#include <stdio.h>
#include <thread>
#include <execution>
#include <algorithm>
#include <random>
#include <chrono>


//#define RENDERDOC_DEBUGGABLE_GLFW3
#ifdef RENDERDOC_DEBUGGABLE_GLFW3
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#endif

//#define RENDERDOC_DEBUG
#ifdef RENDERDOC_DEBUG
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <renderdoc.h>
#endif

namespace rad {

    vk::Instance ComputeFramework::createInstance() {

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

        vk::Device ComputeFramework::createDevice(bool isComputePrior, std::string shaderPath, bool enableAdvancedAcceleration) {
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



#ifdef RENDERDOC_DEBUGGABLE
    static void error_callback(int error, const char* description)
    {
        fprintf(stderr, "Error: %s\n", description);
    }
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
#endif


    TestSort::TestSort(){

#ifdef RENDERDOC_DEBUGGABLE
        GLFWwindow* window = {};

        {
            // init GLFW fast
            glfwSetErrorCallback(error_callback);
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            if (!glfwInit()) exit(EXIT_FAILURE);

            window = glfwCreateWindow(640, 480, "RenderDoc Debuggable", NULL, NULL);
            if (!window) {
                glfwTerminate();
                exit(EXIT_FAILURE);
            }
            glfwSetKeyCallback(window, key_callback);
            glfwMakeContextCurrent(window);
            glfwSwapInterval(1);
        };
#endif

        fw = std::make_shared<ComputeFramework>();
        fw->createInstance();

#if defined(RENDERDOC_DEBUG)
        RENDERDOC_API_1_1_2* rdoc_api = {};

        pRENDERDOC_GetAPI RENDERDOC_GetAPI{};
            #if defined(_WIN32)
                if (HMODULE mod = GetModuleHandleA("renderdoc.dll"))
                {
                    RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
                }
            #else
                if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
                {
                    RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
                }
            #endif
            RENDERDOC_DevicePointer rdoc_device{};
            RENDERDOC_WindowHandle rdoc_window{};
            if (RENDERDOC_GetAPI)
            {
                if (RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api))
                {
                    int major = 0, minor = 0, patch = 0;
                    rdoc_api->GetAPIVersion(&major, &minor, &patch);
                    std::cout << ("Detected RenderDoc API " + std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch)) << std::endl;
                }
                else
                {
                    std::cout << ("Failed to load RenderDoc API") << std::endl;
                }
            }
#endif

        // create radix sort application (RadX C++)
        physicalHelper = std::make_shared<radx::PhysicalDeviceHelper>(fw->getPhysicalDevice(0));
        device = std::make_shared<radx::Device>()->initialize(fw->createDevice(), physicalHelper);
        program = std::make_shared<radx::Radix>(), program->initialize(device);
        radixSort = std::make_shared<radx::Sort<radx::Radix>>(), radixSort->initialize(device, program, elementCount);
        inputInterface = std::make_shared<radx::InputInterface>(device);

        { // sizes of offsets
            keysSize = elementCount * sizeof(uint32_t), valuesSize = elementCount * sizeof(uint32_t);
            keysOffset = 0, valuesOffset = keysOffset + keysSize;
        };

        // get memory size and set max element count
        vk::DeviceSize memorySize = valuesOffset + valuesSize;
		{
			vmaDeviceBuffer = std::make_shared<radx::VmaAllocatedBuffer>(this->device, memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_GPU_ONLY);
			vmaHostBuffer = std::make_shared<radx::VmaAllocatedBuffer>(this->device, memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_GPU_TO_CPU );
			vmaToHostBuffer = std::make_shared<radx::VmaAllocatedBuffer>(this->device, memorySize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);
		};

		// 
		radx::Vector<uint32_t> 
			keysHostVector = radx::Vector<uint32_t>(vmaHostBuffer, elementCount, keysOffset),
			keysToHostVector = radx::Vector<uint32_t>(vmaToHostBuffer, elementCount, keysOffset), // for debugging
			keysDeviceVector = radx::Vector<uint32_t>(vmaDeviceBuffer, elementCount, keysOffset),
			valuesDeviceVector = radx::Vector<uint32_t>(vmaDeviceBuffer, elementCount, valuesOffset),
			keysHostVectorCopy = keysHostVector // for C++ debug
			;

        // on deprecation
        inputInterface->setElementCount(keysDeviceVector.size());
        inputInterface->setKeysBufferInfo(keysDeviceVector);
        inputInterface->setValuesBufferInfo(valuesDeviceVector);

        // build descriptor set
		inputInterface->buildDescriptorSet();


        // random engine
        std::random_device rd;
        std::mt19937_64 eng(rd());
        std::uniform_int_distribution<uint32_t> distr;

		
		// generate random numbers and copy to buffer
		keysHostVector.map();
        for (uint32_t i=0;i< keysHostVector.size();i++) { keysHostVector[i] = distr(eng); };
        //memcpy(keysHostVector->map(), sortedNumbers.data(), keysHostVector->range()); // copy from std::vector

        // command allocation
        vk::CommandBufferAllocateInfo cci{};
        cci.commandPool = fw->getCommandPool();
        cci.commandBufferCount = 3;
        cci.level = vk::CommandBufferLevel::ePrimary;
		std::vector<vk::CommandBuffer> cmdBuffers = vk::Device(*device).allocateCommandBuffers(cci);

        // query pool
        vk::QueryPoolCreateInfo qpi{};
        qpi.queryType = vk::QueryType::eTimestamp;
        qpi.queryCount = 2;
        qpi.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eComputeShaderInvocations;
		auto queryPool = vk::Device(*device).createQueryPool(qpi);

        // generate command
		{
			auto& uploadCmdBuf = cmdBuffers.at(0);
			uploadCmdBuf.begin(vk::CommandBufferBeginInfo());
			uploadCmdBuf.copyBuffer(keysHostVector, keysDeviceVector, { vk::BufferCopy(keysHostVector.offset(), keysDeviceVector.offset(), keysHostVector.range()) }); // copy buffer to host
			commandTransferBarrier(uploadCmdBuf);
			uploadCmdBuf.end();
		};

		{
			auto& sortCmdBuf = cmdBuffers.at(1);
			sortCmdBuf.begin(vk::CommandBufferBeginInfo());
			sortCmdBuf.resetQueryPool(queryPool, 0, 2);
			sortCmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, 0);
			radixSort->command(sortCmdBuf, inputInterface);
			sortCmdBuf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, queryPool, 1);
			commandTransferBarrier(sortCmdBuf);
			sortCmdBuf.end();
		};

		{ // copy to host into dedicated buffer (for debug only)
			auto& downloadCmdBuf = cmdBuffers.at(2);
			downloadCmdBuf.begin(vk::CommandBufferBeginInfo());
			downloadCmdBuf.copyBuffer(keysDeviceVector, keysToHostVector, { vk::BufferCopy(keysDeviceVector.offset(), keysToHostVector.offset(), keysDeviceVector.range()) }); // copy buffer to host
			commandTransferBarrier(downloadCmdBuf);
			downloadCmdBuf.end();
		};

#ifdef RENDERDOC_DEBUG
        if (rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);
#endif

        // submit command
        vk::SubmitInfo sbmi = {};
        sbmi.commandBufferCount = cmdBuffers.size();
		sbmi.pCommandBuffers = cmdBuffers.data();

		// submit commands
        auto fence = fw->getFence();
		fw->getQueue().submit(sbmi, fence);
		vk::Device(*device).waitForFences({ fence }, true, INT32_MAX);
		vk::Device(*device).resetFences({ 1, &fence });

		// get Vulkan API timestamp measure result
        std::array<uint64_t, 2> stamps{};
        vk::Device(*device).getQueryPoolResults(queryPool, 0u, 2u, vk::ArrayProxy<uint64_t>{ 2, &stamps[0] }, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
        double diff = double(stamps[1] - stamps[0])/1e6;
        std::cout << "GPU sort measured in " << diff << "ms" << std::endl;

#ifdef RENDERDOC_DEBUG
        if(rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);
#endif

        // do std sort for comparsion (equalent)
		// for better result's should be work while GPU sorting (after copying host data into device)
        auto start = std::chrono::system_clock::now();
        std::sort(std::execution::par, keysHostVector.map(), keysHostVector.end());
        auto end = std::chrono::system_clock::now();
        std::cout << "CPU sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;

		// get sorted numbers by device (for debug only)
		// used alternate buffer, because std::sort already overriden 'keysHostVector' data 
		std::vector<uint32_t> sortedNumbers(elementCount);
		memcpy(sortedNumbers.data(), keysToHostVector.map(), keysToHostVector.range()); // copy

		// thrust sorting
#ifdef ENABLE_THRUST_BENCHMARK
		this->testSortingThrust();
#endif

        //
        std::cout << "Sorting Finished" << std::endl;
        system("pause");


#ifdef RENDERDOC_DEBUGGABLE
        // debug processing
        while (!glfwWindowShouldClose(window))
        {
            //glfwSwapBuffers(window);
            glfwPollEvents();
        };
        glfwDestroyWindow(window);
        glfwTerminate();
#endif

    };
};
