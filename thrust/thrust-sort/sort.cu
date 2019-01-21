
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
	std::vector<uint32_t> randNumbers(elementCount);
	std::vector<uint32_t> sortedNumbersThrust(elementCount);
    for (uint32_t i=0;i<randNumbers.size();i++) { randNumbers[i] = distr(eng); };

    thrust::device_vector<uint32_t> randNumbersThrust(elementCount);  
    thrust::copy(randNumbers.begin(), randNumbers.end(), randNumbersThrust.begin());
	auto start = std::chrono::system_clock::now();
    thrust::sort(randNumbersThrust.begin(), randNumbersThrust.end());
	auto end = std::chrono::system_clock::now();
	std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;
    thrust::copy(randNumbersThrust.begin(), randNumbersThrust.end(), sortedNumbersThrust.begin());

	system("pause");

	return 0;
};
