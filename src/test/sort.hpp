#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#define VKU_NO_GLFW
#include "radx/radx.hpp"

namespace rad {

    class TestSort : public std::enable_shared_from_this<TestSort> {
        protected:
        std::shared_ptr<radx::Device> device;
        std::shared_ptr<radx::Radix> program;
        std::shared_ptr<radx::Sort<radx::Radix>> radixSort;
        std::shared_ptr<radx::PhysicalDeviceHelper> physicalHelper;

        public:
        TestSort(){
            // TODO: create vkPhysicalDevice and vk::Device

            // create radix sort application (RadX C++)
            physicalHelper = std::make_shared<radx::PhysicalDeviceHelper>(vkphysical);
            device = std::make_shared<radx::Device>()->initialize(vkdevice, physicalHelper);
            program = std::make_shared<radx::Radix>();
            radixSort = std::make_shared<radx::Sort<radx::Radix>>();
            radixSort->initialize(device, program->initialize(device));
            
        }
    };

};
