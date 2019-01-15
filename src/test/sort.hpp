#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

#include "radx/radx.hpp"

namespace rad {

    class TestSort : public std::enable_shared_from_this<TestSort> {
        protected:
        std::shared_ptr<radx::Device> device;
        std::shared_ptr<radx::Radix> program;
        std::shared_ptr<radx::Sort<radx::Radix>> radixSort;

        public:
        TestSort(){
            auto instance = vuh::Instance();
            auto vdevice = instance.devices().at(0);

            device = std::make_shared<radx::Device>()->initialize(vdevice);
            program = std::make_shared<radx::Radix>();
            program->initialize(device);
            

        }
    };

};
