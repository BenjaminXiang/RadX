#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include <chrono>
#include <vector>
#include <random>
#include <iostream>

const size_t elementCount = (1 << 23);

int main() {

    // random engine
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<uint32_t> distr;

    // generate random numbers and copy to buffer
    thrust::host_vector<uint32_t> randNumbers(elementCount);
    thrust::device_vector<uint32_t> keysDev(elementCount);
    thrust::device_vector<uint32_t> keysDevBackup(elementCount);
    thrust::device_vector<uint32_t> valuesDev(elementCount);
    thrust::host_vector<uint32_t> sortedNumbersThrust(elementCount);
    std::vector<uint32_t> sortedNumbers(elementCount);
    for (uint32_t i = 0; i < randNumbers.size(); i++) { randNumbers[i] = i; };
    std::shuffle(randNumbers.begin(), randNumbers.end(), eng);


    // command and execution
    cudaDeviceSynchronize();
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float totalTime = 0;

    thrust::copy(randNumbers.begin(), randNumbers.end(), keysDevBackup.begin());

    cudaDeviceSynchronize();
    cudaEventRecord(start_event, 0);
    thrust::copy(keysDevBackup.begin(), keysDevBackup.end(), keysDev.begin());
    thrust::stable_sort(keysDev.begin(), keysDev.end());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&totalTime, start_event, stop_event);
    cudaDeviceSynchronize();

    thrust::copy(keysDev.begin(), keysDev.end(), sortedNumbersThrust.begin());
    cudaDeviceSynchronize();

    // copy from device to host (finally)
    std::copy(sortedNumbersThrust.begin(), sortedNumbersThrust.end(), sortedNumbers.begin()); // on-host copying (for debugging)
    std::cout << "Thrust sort measured in " << double(totalTime) << "ms" << std::endl;
    //std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;
    system("pause");

    return 0;
};
