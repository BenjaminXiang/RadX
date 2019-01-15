#pragma once 
#include "radx_core.hpp"

// TODO: 
// - getting features and properties
// - getting vendor naming
// - detecting what is GPU

namespace radx {
    class Device : public std::enable_shared_from_this<Device> {
        protected:
        
        public:
        std::shared_ptr<Device> initialize(vuh::Device& vdevice){


            return shared_from_this();
        };
    };
};
