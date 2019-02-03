#include <stdio.h>
#include <cstdint>


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
#include <algorithm>

const size_t elementCount = 1 << 17;//(1 << 23);

namespace radx {
    const uint32_t RADICES = 256u;
    const uint32_t VECSIZE = 4u;
    const uint32_t BIT_CNT = 8u;
    const uint32_t WAVE_SIZE = 32u;
    const uint32_t WG_COUNT = 1u;

    //#define validAddress cub::BFE(validAddressL[waveID],laneID,1u)


    template <typename T>
    __host__ __device__ static inline auto sgn(const T& val) { return (T(0) < val) - (val < T(0)); }

    template<class T = uint64_t>
    __host__ __device__ static inline T tiled(const T& sz, const T& gmaxtile) {
        // return (int32_t)ceil((double)sz / (double)gmaxtile);
        return sz <= 0 ? 0 : (sz / gmaxtile + sgn(sz % gmaxtile));
    }


    struct blocks_info { uint32_t count, size, limit, wkoffset; };
    __host__ blocks_info get_blocks_info(const uint32_t& n) {
        const uint32_t
            block_tile = WAVE_SIZE << 2,
            block_size_per_work = tiled(n, WG_COUNT),
            block_size = tiled(block_size_per_work, block_tile) * block_tile,
            block_offset = 0u,//block_size * gl_WorkGroupID.x,
            block_limit = block_offset + block_size,
            block_count = tiled(block_size, block_tile);

        return blocks_info{ block_count, block_size, n, 0u };
    };


//#define addressW addressL[threadIdx.x]
//#define keyW keysL[waveID][laneID]

    // 
    __global__ void Counting(
        const uint32_t P, const uint32_t bcount, const uint32_t bsize, const uint32_t limit, 
        uint32_t * countsBuf, const uint32_t * keysStorage
    ) {
        const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();
        __shared__ uint32_t localCounts[RADICES];
        __shared__ uint32_t addressL[VECSIZE*WAVE_SIZE];
        __shared__ uint8_t keysL[WAVE_SIZE][VECSIZE];
        __shared__ uint32_t validAddressL[VECSIZE];

        for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
            localCounts[r+threadIdx.x] = 0u;
        };

        auto& keyW = keysL[waveID][laneID];
        auto& addressW = addressL[threadIdx.x];
        __syncthreads();

        addressW = threadIdx.x + (bsize*blockIdx.x);
        for (uint32_t t=0;t<bcount;t++) {
            bool predicate = addressW < limit;
            keyW = predicate ? cub::BFE(keysStorage[addressW],BIT_CNT*P,BIT_CNT) : 0xFFu;
            validAddressL[waveID] = __ballot_sync(__activemask(), predicate);

            // counting in SM with atomics
            uint32_t prtmask = __match_any_sync(__activemask(), keyW)&validAddressL[waveID]; {
                uint32_t leader = __ffs(prtmask) - 1u;
                uint32_t cnt = 0u; if (laneID == leader) {cnt = atomicAdd(&localCounts[keyW], __popc(prtmask));}; //cnt = __shfl_sync(prtmask, cnt, leader);
            };

            addressW += blockDim.x;
        };
        __syncthreads();

        for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
            countsBuf[blockIdx.x * RADICES + (r+threadIdx.x)] = localCounts[r+threadIdx.x];
        };
    };

    // 
    __global__ void Scattering(
        const uint32_t P, const uint32_t bcount, const uint32_t bsize, const uint32_t limit, 
        const uint32_t * partitionsBuf, const uint32_t * keysStorage, uint32_t * keysBackup
    ) {
        const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();
        __shared__ uint32_t localCounts[RADICES], localPartitions[RADICES];
        __shared__ uint32_t addressL[VECSIZE*WAVE_SIZE];
        __shared__ uint8_t keysL[WAVE_SIZE][VECSIZE];
        __shared__ uint32_t prefixL[WAVE_SIZE][VECSIZE], sumL[WAVE_SIZE][VECSIZE], flaneL[WAVE_SIZE][VECSIZE];
        __shared__ uint32_t validAddressL[VECSIZE];


        for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
            localPartitions[r+threadIdx.x] = partitionsBuf[blockIdx.x * RADICES + (r+threadIdx.x)], localCounts[r+threadIdx.x] = 0u;
        };
        __syncthreads();

        auto& cnt = sumL[waveID][laneID], prefix = prefixL[waveID][laneID];
        auto& keyW = keysL[waveID][laneID];
        auto& addressW = addressL[threadIdx.x];
        auto& leader = flaneL[waveID][laneID];

        addressW = threadIdx.x + (bsize*blockIdx.x);
        for (uint32_t t=0;t<bcount;t++) {
            bool predicate = addressW < limit;
            keyW = predicate ? cub::BFE(keysStorage[addressW], BIT_CNT*P, BIT_CNT) : 0xFFu;
            validAddressL[waveID] = __ballot_sync(__activemask(), predicate);

            uint32_t prtmask = __match_any_sync(__activemask(), keyW)&validAddressL[waveID];
            { prefix = __popc(prtmask & cub::LaneMaskLt()), cnt = __popc(prtmask), leader = __ffs(prtmask) - 1u; };
            __syncthreads();

            // counting in SM with atomics
            if (waveID == 0u) for (uint32_t w=0;w<VECSIZE;w++) { uint32_t sumt = 0u; 
                auto& cnt = sumL[w][laneID]; 
                auto& prefix = prefixL[w][laneID];
                auto& keyW = keysL[w][laneID];
                auto& leader = flaneL[w][laneID];

                if (laneID == leader) {sumt = atomicAdd(&localCounts[keyW], cnt);}; 
                if (leader != -1u) prefix += __shfl_sync(prtmask, sumt, leader);
            };
            __syncthreads();
        
            if (predicate) keysBackup[ localPartitions[keyW] + prefix ] = keysStorage[addressW];
            __syncwarp();
            addressW += blockDim.x;
        };
    

    };


    // TODO: development with CUB
    __global__ void Partition(const uint32_t * countsBuf, uint32_t * partitionsBuf) {
        const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();

        __shared__ uint32_t localCounts[RADICES];
        typedef cub::WarpScan<uint32_t> WarpScan;
        __shared__ typename WarpScan::TempStorage temp_storage;
        for (uint32_t r=0;r<RADICES;r+=blockDim.x) { localCounts[r+threadIdx.x] = 0u; };

        __syncthreads();

        for (uint32_t rk=0u;rk<RADICES;rk+=32u) { uint32_t radice = rk + waveID;
            for (uint32_t gp=0u;gp<WG_COUNT;gp+=WAVE_SIZE) { uint32_t workgroup = gp+laneID;
            
                // validate masks
                bool predicate = workgroup < WG_COUNT && radice < RADICES;
                uint32_t mask = __ballot_sync(__activemask(), predicate);
                uint32_t mostb = 31u - __clz(mask); // MSB

                // 
                uint32_t cnt = predicate ? countsBuf[workgroup*RADICES+radice] : 0u, scan = 0u;
            
                // prefix scan and sum
                WarpScan(temp_storage).ExclusiveSum(cnt, scan);
                uint32_t sum = __shfl_sync(mask, cnt+scan, mostb);

                // complete phase
                uint32_t pref = 0u; if (laneID == 0u) { pref = atomicAdd(&localCounts[radice], sum); };
                if (predicate) { partitionsBuf[workgroup*RADICES+radice] = __shfl_sync(mask, pref, 0) + scan; };
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
                uint32_t cnt = predicate ? localCounts[radice] : 0u, scan = 0u;
            
                // prefix scan and sum
                WarpScan(temp_storage).ExclusiveSum(cnt, scan);
                uint32_t sum = __shfl_sync(mask, cnt+scan, mostb);

                // complete phase
                uint32_t pref = 0u; if (laneID == 0u) { pref = partsum, partsum += sum; };
                if (predicate) { partitionsBuf[workgroup*RADICES+radice] += __shfl_sync(mask, pref, 0) + scan; };
            };
        };

        __syncthreads();
    };

    // from second, used only begin iterator
    __host__ void sort(thrust::device_vector<uint32_t>& keysStore, thrust::device_vector<uint32_t>& keysBackup) {

        uint32_t *keyStorePtr = thrust::raw_pointer_cast(&keysStore[0]), *keyBackupPtr = thrust::raw_pointer_cast(&keysBackup[0]);
        uint32_t *histogramMem;
        cudaMalloc(&histogramMem, RADICES*WG_COUNT * sizeof(uint32_t) * 2);

        blocks_info bclk = get_blocks_info(uint32_t(keysStore.size()));
        uint32_t *countMem = histogramMem, *partitionMem = histogramMem + (RADICES*WG_COUNT);

        for (uint32_t p = 0u; p < 2u; p++) {
            Counting << < WG_COUNT, VECSIZE*WAVE_SIZE >> > (
                p, bclk.count, bclk.size, bclk.limit,
                countMem, keyStorePtr
                );
            cudaDeviceSynchronize();
            Partition << < 1u, 1024u >> > (countMem, partitionMem);
            cudaDeviceSynchronize();
            Scattering << < WG_COUNT, VECSIZE*WAVE_SIZE >> > (
                p, bclk.count, bclk.size, bclk.limit,
                partitionMem, keyStorePtr, keyBackupPtr
                );
            cudaDeviceSynchronize();
            std::swap(keyStorePtr, keyBackupPtr);
        };
        cudaDeviceSynchronize();
        //cudaMemcpy(keyStorePtr, keyBackupPtr, sizeof(uint32_t) * keysStore.size(), cudaMemcpyDeviceToDevice);
        //thrust::copy(keysBackup.begin(), keysBackup.end(), keysStore.begin());

        //cudaFree(histogramMem);
    }
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

    thrust::copy(randNumbers.begin(), randNumbers.end(), keysDev.begin());
    cudaDeviceSynchronize();
    cudaEventRecord(start_event, 0);
    //thrust::copy(keysDevBackup.begin(), keysDevBackup.end(), keysDev.begin());
    //thrust::stable_sort(keysDev.begin(), keysDev.end());

    radx::sort(keysDev, keysDevBackup);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&totalTime, start_event, stop_event);
    cudaDeviceSynchronize();

    thrust::copy(keysDev.begin(), keysDev.end(), sortedNumbersThrust.begin());
    cudaDeviceSynchronize();

    // copy from device to host (finally)
    thrust::copy(sortedNumbersThrust.begin(), sortedNumbersThrust.end(), sortedNumbers.begin()); // on-host copying (for debugging)
    std::cout << "Thrust sort measured in " << double(totalTime) << "ms" << std::endl;
    //std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;

    return 0;
};
