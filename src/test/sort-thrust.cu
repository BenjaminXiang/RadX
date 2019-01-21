#include "sort.hpp"

#ifdef ENABLE_THRUST_BENCHMARK
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#endif

namespace rad {

    void TestSort::testSortingThrust()(){
#ifdef ENABLE_THRUST_BENCHMARK
        sortedNumbersThrust.resize(elementCount);
        thrust::device_vector<uint32_t> randNumbersThrust(elementCount);  
        thrust::copy(randNumbers.begin(), randNumbers.end(), randNumbersThrust.begin());
        thrust::sort(randNumbersThrust.begin(), randNumbersThrust.end());

		auto start = std::chrono::system_clock::now();
		thrust::copy(randNumbers.begin(), randNumbers.end(), randNumbersThrust.begin());
		auto end = std::chrono::system_clock::now();
		std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;

        thrust::copy(randNumbersThrust.begin(), randNumbersThrust.end(), sortedNumbersThrust.begin());
#endif
    };

};
