#pragma once 
#include "radx_core.hpp"
#include "radx_device.hpp"

namespace radx {
    namespace paths {
        static inline const std::array<std::string, 6> pathNames{ "universal", "amd", "nvidia", "intel", "vega", "turing" };

        static inline constexpr const auto permutation = "radix/permutation.comp";
        static inline constexpr const auto partition = "radix/partition.comp";
        static inline constexpr const auto counting = "radix/counting.comp";
        static inline constexpr const auto scattering = "radix/scattering.comp";
        static inline constexpr const auto indiction = "radix/indiction.comp";

        static inline const auto getCorrectPath(const std::string& fpath = "", const radx::Vendor& vendor = radx::Vendor::NV_TURING, const std::string& directory = "./intrusive") {
            return (directory + "/" + pathNames[vendor] + "/" + fpath + ".spv");
        };
    };
};
