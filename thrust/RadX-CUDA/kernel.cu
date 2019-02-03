#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include <cub/cub.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/sort.h>
#include <thrust/copy.h>

#include <chrono>
#include <vector>
#include <random>
#include <iostream>

const size_t elementCount = (1 << 23);

const uint32_t RADICES = 256u;
const uint32_t VECSIZE = 4u;
const uint32_t BIT_CNT = 8u;
const uint32_t WAVE_SIZE = 32u;

#define validAddress cub::BFE(validAddressL[waveID],laneID,1u)


// 
__global__ void Counting(const uint32_t P, const uint32_t bcount, const uint32_t limit, const uint32_t * keysStorage, uint32_t * countsBuf) {
    const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();
    __shared__ uint32_t localCounts[RADICES];
    __shared__ uint32_t addressL[VECSIZE*WAVE_SIZE];
    __shared__ uint8_t keysL[WAVE_SIZE][VECSIZE];
    __shared__ uint32_t validAddressL[VECSIZE];

    for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
        localCounts[r+threadIdx.x] = 0u;
    };
    addressL[threadIdx.x] = threadIdx.x;
    _syncthreads();

    uint32_t& address = addressL[threadIdx.x];
    uint8_t& key = keysL[waveID][laneID];
    for (uint32_t t=0;t<bcount;t++) {
        key = cub::BFE(keysStorage[address],BIT_CNT*P,BIT_CNT);
        validAddressL[waveID] = __ballot_sync(__activemask(), address < limit);

        // counting in SM with atomics
        int pred; uint32_t prtmask = __match_all_sync(__activemask(), key, &pred)&validAddressL[waveID]; {
            uint32_t leader = __ffs(prtmask) – 1;
            uint32_t cnt = 0u; if (laneID == leader) {cnt = atomicAdd(localCounts[key],__popc(prtmask));}; //cnt = __shfl_sync(prtmask, cnt, leader);
        };

        address += blockDim.x;
    };
    _syncthreads();

    for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
        countsBuf[blockIdx.x * RADICES + (r+threadIdx.x)] = localCounts[r+threadIdx.x];
    };
};

// 
__global__ void Scattering(const uint32_t P, const uint32_t bcount, const uint32_t limit, const uint32_t * keysStorage, const uint32_t * partitionsBuf, uint32_t * keysBackup) {
    const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();
    __shared__ uint32_t localCounts[RADICES], localPartitions[RADICES];
    __shared__ uint32_t addressL[VECSIZE*WAVE_SIZE];
    __shared__ uint8_t keysL[WAVE_SIZE][VECSIZE];
    __shared__ uint32_t prefixL[WAVE_SIZE][VECSIZE], sumL[WAVE_SIZE][VECSIZE];
    __shared__ uint32_t validAddressL[VECSIZE];

    for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
        localPartitions[r+threadIdx.x] = partitionsBuf[blockIdx.x * RADICES + (r+threadIdx.x)], localCounts[r+threadIdx.x] = 0u;
    };
    addressL[threadIdx.x] = threadIdx.x;
    _syncthreads();


    uint32_t& address = addressL[threadIdx.x];
    uint8_t& key = keysL[waveID][laneID];
    uint32_t& cnt = sumL[waveID][laneID], prefix = prefixL[waveID][laneID];

    for (uint32_t t=0;t<bcount;t++) {
        key = cub::BFE(keysStorage[address],BIT_CNT*P,BIT_CNT);
        validAddressL[waveID] = __ballot_sync(__activemask(), address < limit);
        
        int pred; uint32_t prtmask = __match_all_sync(__activemask(), key, &pred)&validAddressL[waveID];
        { prefix = __popc(mask & __lanemask_lt()), cnt = __popc(prtmask); };
        _syncthreads();

        // counting in SM with atomics
        if (waveID == 0u) for (uint32_t w=0;w<VECSIZE;w++) { uint32_t sumt = 0u; uint32_t& cnt = sumL[w][laneID], prefix = prefixL[w][laneID];
            uint32_t leader = __ffs(prtmask) – 1;
            if (laneID == leader) {sumt = atomicAdd(localCounts[key], cnt);}; prefix += __shfl_sync(prtmask, sumt, leader);
        };
        _syncthreads();
        
        if (validAddress) keysBackup[ localPartitions[key] + prefix ] = keysStorage[address];
        _syncwarp();
        address += blockDim.x;
    };
    

};


// TODO: development with CUB
__global__ void Partition(const uint32_t * countsBuf, uint32_t * partitionsBuf) {
    const uint32_t& laneID = get_lane_id(), waveID = get_warp_id();


};



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

    return 0;
};
