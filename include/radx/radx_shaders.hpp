#pragma once 
#include "radx_core.hpp"
#include "radx_device.hpp"

namespace radx {
    namespace paths {
        static inline const std::array<std::string, 6> pathNames{ "universal", "amd", "nvidia", "intel", "vega", "turing" };

        static inline constexpr const auto permute = "radix/permute.comp";
        static inline constexpr const auto workload = "radix/pfx-work.comp";
        static inline constexpr const auto histogram = "radix/histogram.comp";
        static inline constexpr const auto copyhack = "radix/copyhack.comp";
        static inline constexpr const auto transposer = "radix/transposer.comp";

        static inline const auto getCorrectPath(const std::string& fpath = "", const radx::Vendor& vendor = radx::Vendor::NV_TURING, const std::string& directory = "./intrusive") {
            return (directory + "/" + pathNames[vendor] + "/" + fpath + ".spv");
        };
    };
};
