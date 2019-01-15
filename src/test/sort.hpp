#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#define VKU_NO_GLFW
#include "radx/radx.hpp"
#include "vku/vku_framework.hpp"

namespace rad {

    class TestSort : public std::enable_shared_from_this<TestSort> {
        protected:
        std::shared_ptr<radx::Device> device;
        std::shared_ptr<radx::Radix> program;
        std::shared_ptr<radx::Sort<radx::Radix>> radixSort;
        std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper;

        public:
        TestSort(){
            vku::Framework fw{"Radix sort test application"};
            if (!fw.ok()) { std::cout << "Framework creation failed" << std::endl; exit(1); }

            // Get a device from the demo framework.
            auto vdevice = fw.device();
            auto cache = fw.pipelineCache();
            auto descriptorPool = fw.descriptorPool();
            auto memprops = fw.memprops();
            auto physical = fw.physicalDevice();

            // create radix sort application (RadX C++)
            physicalHelper = std::make_shared<radx::PhysicalDeviceHelper>(physical);
            device = std::make_shared<radx::Device>()->initialize(vdevice, physicalHelper);
            program = std::make_shared<radx::Radix>();
            radixSort = std::make_shared<radx::Sort<radx::Radix>>();
            radixSort->initialize(device, program->initialize(device));
            
        }
    };

};
