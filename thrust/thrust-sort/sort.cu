
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <chrono>
#include <vector>
#include <random>
#include <iostream>

const size_t elementCount = (2 << 22);

int main(){

    // random engine
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<uint32_t> distr;

	// generate random numbers and copy to buffer
	thrust::host_vector<uint32_t> randNumbers(elementCount);
	thrust::device_vector<uint32_t> randNumbersDev(elementCount);  
	thrust::host_vector<uint32_t> sortedNumbersThrust(elementCount);
	std::vector<uint32_t> sortedNumbers(elementCount);
    for (uint32_t i=0;i<randNumbers.size();i++) { randNumbers[i] = distr(eng); };
	//auto start = std::chrono::system_clock::now();

	// command and execution 
	auto start = std::chrono::system_clock::now();
	thrust::copy(randNumbers.begin(), randNumbers.end(), randNumbersDev.begin());
    thrust::sort(randNumbersDev.begin(), randNumbersDev.end());
	thrust::copy(randNumbersDev.begin(), randNumbersDev.end(), sortedNumbersThrust.begin());
	auto end = std::chrono::system_clock::now();

	// copy from device to host (finally)
	std::copy(sortedNumbersThrust.begin(), sortedNumbersThrust.end(), sortedNumbers.begin()); // on-host copying (for debugging)
	std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;
	system("pause");

	return 0;
};
