#include <stdio.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>

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
const uint32_t WG_COUNT = 128u;

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
    __syncthreads();

    uint32_t& address = addressL[threadIdx.x];
    uint8_t& key = keysL[waveID][laneID];
    for (uint32_t t=0;t<bcount;t++) {
        key = cub::BFE(keysStorage[address],BIT_CNT*P,BIT_CNT);
        validAddressL[waveID] = __ballot_sync(__activemask(), address < limit);

        // counting in SM with atomics
        int pred; uint32_t prtmask = __match_all_sync(__activemask(), key, &pred)&validAddressL[waveID]; {
            uint32_t leader = __ffs(prtmask) - 1u;
            uint32_t cnt = 0u; if (laneID == leader) {cnt = atomicAdd(&localCounts[key], __popc(prtmask));}; //cnt = __shfl_sync(prtmask, cnt, leader);
        };

        address += blockDim.x;
    };
    __syncthreads();

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
    __syncthreads();


    uint32_t& address = addressL[threadIdx.x];
    uint8_t& key = keysL[waveID][laneID];
    uint32_t& cnt = sumL[waveID][laneID], prefix = prefixL[waveID][laneID];

    for (uint32_t t=0;t<bcount;t++) {
        key = cub::BFE(keysStorage[address],BIT_CNT*P,BIT_CNT);
        validAddressL[waveID] = __ballot_sync(__activemask(), address < limit);
        
        int pred; uint32_t prtmask = __match_all_sync(__activemask(), key, &pred)&validAddressL[waveID];
        { prefix = __popc(prtmask & cub::LaneMaskLt()), cnt = __popc(prtmask); };
        __syncthreads();

        // counting in SM with atomics
        if (waveID == 0u) for (uint32_t w=0;w<VECSIZE;w++) { uint32_t sumt = 0u; uint32_t& cnt = sumL[w][laneID], prefix = prefixL[w][laneID];
            uint32_t leader = __ffs(prtmask) - 1u;
            if (laneID == leader) {sumt = atomicAdd(&localCounts[key], cnt);}; prefix += __shfl_sync(prtmask, sumt, leader);
        };
        __syncthreads();
        
        if (validAddress) keysBackup[ localPartitions[key] + prefix ] = keysStorage[address];
        __syncwarp();
        address += blockDim.x;
    };
    

};


// TODO: development with CUB
__global__ void Partition(const uint32_t * countsBuf, uint32_t * partitionsBuf) {
    const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();

    __shared__ uint32_t localCounts[RADICES];
    typedef cub::WarpScan<uint32_t> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[1];
    for (uint32_t r=0;r<RADICES;r+=blockDim.x) { localCounts[r+threadIdx.x] = 0u; };

    for (uint32_t rk=0u;rk<RADICES;rk+=32u) { uint32_t radice = rk + waveID;
        for (uint32_t gp=0u;gp<WG_COUNT;gp+=WAVE_SIZE) { uint32_t workgroup = gp+laneID;
            
            // validate masks
            bool predicate = workgroup < WG_COUNT && radice < RADICES;
            uint32_t mask = __ballot_sync(__activemask(), predicate);
            uint32_t mostb = 31 - __clz(mask); // MSB

            // 
            uint32_t cnt = predicate ? countsBuf[workgroup*RADICES+radice] : 0u, scan = 0u;
            
            // prefix scan and sum
            WarpScan(temp_storage[0]).ExclusiveSum(cnt, scan);
            uint32_t sum = __shfl_sync(mask, cnt+scan, mostb);

            // complete phase
            uint32_t pref = 0u; if (laneID == 0u) { pref = atomicAdd(&localCounts[radice], sum); };
            if (predicate) { partitionsBuf[workgroup*RADICES+radice] = pref + scan; };
        };
    };

    __syncthreads();

    for (uint32_t gp=0u;gp<WG_COUNT;gp+=32u) { uint32_t workgroup = gp+waveID;
        uint32_t partsum = 0u;
        for (uint32_t rk=0u;rk<RADICES;rk+=WAVE_SIZE) { uint32_t radice = rk+laneID;

            // validate masks
            bool predicate = workgroup < WG_COUNT && radice < RADICES;
            uint32_t mask = __ballot_sync(__activemask(), predicate);
            uint32_t mostb = 31 - __clz(mask); // MSB

            // 
            typedef cub::WarpScan<uint32_t> WarpScan;
            uint32_t cnt = predicate ? localCounts[radice] : 0u, scan = 0u;
            
            // prefix scan and sum
            WarpScan(temp_storage[0]).ExclusiveSum(cnt, scan);
            uint32_t sum = __shfl_sync(mask, cnt+scan, mostb);

            // complete phase
            uint32_t pref = 0u; if (laneID == 0u) { pref = partsum, partsum += sum; };
            if (predicate) { partitionsBuf[workgroup*RADICES+radice] += pref + scan; };
        };
    };
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
