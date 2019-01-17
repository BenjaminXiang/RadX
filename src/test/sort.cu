#include "sort.hpp"

#ifdef ENABLE_THRUST_BENCHMARK
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#endif

namespace rad {

#ifdef ENABLE_THRUST_BENCHMARK
    void TestSort::testSortingThrust(){
        sortedNumbersThrust.resize(elementCount);
        thrust::device_vector<uint32_t> randNumbersThrust(elementCount);  
        thrust::copy(randNumbers.begin(), randNumbers.end(), randNumbersThrust.begin());
        thrust::sort(randNumbersThrust.begin(), randNumbersThrust.end());
        thrust::copy(randNumbersThrust.begin(), randNumbersThrust.end(), sortedNumbersThrust.begin());
    };
#endif

};