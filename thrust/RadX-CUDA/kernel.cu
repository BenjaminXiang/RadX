#include <stdio.h>
#include <cstdint>


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>
#include <math.h>

#include <cub/cub.cuh>

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/generate.h>
//#include <thrust/sort.h>
//#include <thrust/copy.h>

#include <chrono>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>



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

        for (uint32_t r=0;r<RADICES;r+=blockDim.x) { const uint32_t radice = r + threadIdx.x; localCounts[radice] = 0u; };

        auto& keyW = keysL[waveID][laneID];
        auto& addressW = addressL[threadIdx.x];
        
        addressW = threadIdx.x + (bsize*blockIdx.x); __syncthreads();

        for (uint32_t t=0;t<bcount;t++) {
            bool predicate = addressW < limit;

            // get element 
            const uint32_t& keyM = keysStorage[addressW];
            keyW = predicate ? cub::BFE(keyM, BIT_CNT*P, BIT_CNT) : 0xFFu;

            // predicate mask in local group
            const uint32_t pmask_ = __ballot_sync(__activemask(), predicate);
            if (laneID == 0u) validAddressL[waveID] = pmask_;
            const uint32_t& pmask = validAddressL[waveID];

            // counting in SM with atomics
            {
                int pred = false; uint32_t prtmask = __match_all_sync(__activemask(), keyW, &pred);
                uint32_t cnt = __popc(prtmask & pmask), leader = __ffs(prtmask & pmask) - 1u;
                if (laneID == leader) {atomicAdd(&localCounts[keyW], cnt);}; //cnt = __shfl_sync(prtmask, cnt, leader);
            };

            addressW += blockDim.x;
        };
        __syncthreads();

        for (uint32_t r=0;r<RADICES;r+=blockDim.x) { const uint32_t radice = r + threadIdx.x;
            countsBuf[blockIdx.x * RADICES + radice] = localCounts[radice];
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


        for (uint32_t r=0;r<RADICES;r+=blockDim.x) { const uint32_t radice = r + threadIdx.x;
            localPartitions[radice] = partitionsBuf[blockIdx.x * RADICES + radice], localCounts[radice] = 0u;
        };

        auto& cnt = sumL[waveID][laneID], prefix = prefixL[waveID][laneID];
        auto& keyW = keysL[waveID][laneID];
        auto& addressW = addressL[threadIdx.x];
        auto& leader = flaneL[waveID][laneID];

        addressW = threadIdx.x + (bsize*blockIdx.x); __syncthreads();

        for (uint32_t t=0;t<bcount;t++) {
            bool predicate = addressW < limit;

            // get element 
            const uint32_t& keyM = keysStorage[addressW];
            keyW = predicate ? cub::BFE(keyM, BIT_CNT*P, BIT_CNT) : 0xFFu;

            // predicate mask in local group
            const uint32_t pmask_ = __ballot_sync(__activemask(), predicate);
            if (laneID == 0u) validAddressL[waveID] = pmask_;
            const uint32_t& pmask = validAddressL[waveID];

            // counting in SM with atomics
            if (waveID == 0u) for (uint32_t w=0;w<VECSIZE;w++) {
                auto& keyW = keysL[w][laneID];

                { // 
                    int pred = false; uint32_t prtmask = __match_all_sync(__activemask(), keyW, &pred);
                    uint32_t prefix = __popc(prtmask & pmask & cub::LaneMaskLt()), cnt = __popc(prtmask & pmask), leader = __ffs(prtmask & pmask) - 1u;

                    // 
                    uint32_t sumt = 0u; if (laneID == leader) { sumt = localCounts[keyW]; localCounts[keyW] += cnt; };
                    prefix += __shfl_sync(__activemask(), sumt, leader);
                };
            };
            __syncthreads();
            
            if (predicate) {
                const uint32_t& partition = localPartitions[keyW];
                keysBackup[partition + prefix] = keyM;
            };
            __syncwarp();
            addressW += blockDim.x;
        };
        __syncthreads();
    

    };


    // TODO: development with CUB
    __global__ void Partition(const uint32_t * countsBuf, uint32_t * partitionsBuf) {
        const uint32_t& laneID = cub::LaneId(), waveID = cub::WarpId();
        const uint32_t waveCount = 8u;

        typedef cub::WarpScan<uint32_t> WarpScan;
        typedef cub::WarpReduce<uint32_t> WarpReduce;
        __shared__ typename WarpScan::TempStorage scanStorage[waveCount];
        __shared__ typename WarpReduce::TempStorage reduceStorage[waveCount];

        __shared__ uint32_t localCounts[RADICES];
        for (uint32_t r=0;r<RADICES;r+=blockDim.x) {
            const uint32_t radice = r + threadIdx.x;
            if (radice < RADICES) localCounts[radice] = 0u;
        };

        __syncthreads();

        for (uint32_t rk=0u;rk<RADICES;rk+= waveCount) { uint32_t radice = rk + waveID;
            for (uint32_t gp=0u;gp<WG_COUNT;gp+=WAVE_SIZE) { uint32_t workgroup = gp+laneID;
            
                // validate masks
                const bool predicate = workgroup < WG_COUNT && radice < RADICES;
                const uint32_t activem = __activemask();
                const uint32_t mask = __ballot_sync(activem, predicate);
                const uint32_t mostb = 31u - __clz(mask); // MSB

                // prefix scan and sum
                const uint32_t cnt = predicate ? countsBuf[workgroup*RADICES + radice] : 0u;
                uint32_t scan = 0u; WarpScan(scanStorage[waveID]).ExclusiveSum(cnt, scan);
                uint32_t sum = 0u; WarpReduce(reduceStorage[waveID]).Sum(cnt, sum); //sum = __shfl_sync(activem, cnt + scan, mostb);

                // complete phase
                uint32_t pref = 0u; if (laneID == 0u) { pref = localCounts[radice]; localCounts[radice] += sum; }; pref = __shfl_sync(activem, pref, 0);
                if (predicate) { partitionsBuf[workgroup*RADICES+radice] = pref + scan; };
            };
        };

        __syncthreads();

        for (uint32_t gp=0u;gp<WG_COUNT;gp+= waveCount) { uint32_t workgroup = gp+waveID; uint32_t partsum = 0u;
            for (uint32_t rk=0u;rk<RADICES;rk+=WAVE_SIZE) { uint32_t radice = rk+laneID;

                // validate masks
                const bool predicate = workgroup < WG_COUNT && radice < RADICES;
                const uint32_t activem = __activemask();
                const uint32_t mask = __ballot_sync(activem, predicate);
                const uint32_t mostb = 31u - __clz(mask); // MSB

                // prefix scan and sum
                const uint32_t cnt = predicate ? localCounts[radice] : 0u;
                uint32_t scan = 0u; WarpScan(scanStorage[waveID]).ExclusiveSum(cnt, scan);
                uint32_t sum = 0u; WarpReduce(reduceStorage[waveID]).Sum(cnt, sum); //sum = __shfl_sync(activem, cnt + scan, mostb);

                // complete phase
                uint32_t pref = 0u; if (laneID == 0u) { pref = partsum, partsum += sum; }; pref = __shfl_sync(activem, pref, 0);
                if (predicate) { partitionsBuf[workgroup*RADICES+radice] += pref + scan; };
            };
        };

        __syncthreads();
    };

//namespace radx {
    // from second, used only begin iterator
    void sort(uint32_t *keysStorePtr, uint32_t *keysBackupPtr, size_t size) {

        //uint32_t *keysStorePtr = thrust::raw_pointer_cast(&keysStore[0]), *keysBackupPtr = thrust::raw_pointer_cast(&keysBackup[0]);
        uint32_t *histogramMem;//, *keysBackupPtr;
        cudaMalloc(&histogramMem, RADICES*WG_COUNT * sizeof(uint32_t) * 2);
        //cudaMalloc(&keysBackupPtr, keysStore.size()*sizeof(uint32_t));

        blocks_info bclk = get_blocks_info(uint32_t(size));
        uint32_t *countMem = histogramMem, *partitionMem = histogramMem + (RADICES*WG_COUNT);

        //cudaDeviceSynchronize();
        for (uint32_t p = 0u; p < 1u; p++) {
            Counting<<< WG_COUNT, VECSIZE*WAVE_SIZE>>>(
                p, bclk.count, bclk.size, bclk.limit,
                countMem, keysStorePtr
                );
            //cudaDeviceSynchronize();
            Partition<<< 1u, 256u >>>(countMem, partitionMem);
            //cudaDeviceSynchronize();
            Scattering<<< WG_COUNT, VECSIZE*WAVE_SIZE>>>(
                p, bclk.count, bclk.size, bclk.limit,
                partitionMem, keysStorePtr, keysBackupPtr
                );
            //cudaDeviceSynchronize();
            std::swap(keysStorePtr, keysBackupPtr);
        };
        cudaDeviceSynchronize();
        cudaMemcpy(keysBackupPtr, keysStorePtr, sizeof(uint32_t) * size, cudaMemcpyDeviceToDevice);

    }
};


const size_t elementCount = 1 << 8;//(1 << 23);

int main() {

    // random engine
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<uint32_t> distr;

    // generate random numbers and copy to buffer
    //thrust::host_vector<uint32_t> randNumbers(elementCount);
    //thrust::device_vector<uint32_t> keysDev(elementCount);
    //thrust::device_vector<uint32_t> keysDevBackup(elementCount);
    //thrust::device_vector<uint32_t> valuesDev(elementCount);
    //thrust::host_vector<uint32_t> sortedNumbersThrust(elementCount);


    std::vector<uint32_t> randNumbers(elementCount);
    std::vector<uint32_t> sortedNumbers(elementCount);
    for (uint32_t i = 0; i < randNumbers.size(); i++) { randNumbers[i] = i; };
    //std::shuffle(randNumbers.begin(), randNumbers.end(), eng);


    // cuda malloc
    uint32_t *keysDev, *keysDevBackup;
    cudaMalloc(&keysDev, elementCount*sizeof(uint32_t));
    cudaMalloc(&keysDevBackup, elementCount * sizeof(uint32_t));

    // command and execution
    cudaDeviceSynchronize();
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    float totalTime = 0;

    //thrust::copy(randNumbers.begin(), randNumbers.end(), keysDev.begin());
    cudaMemcpy(keysDev, randNumbers.data(), sizeof(uint32_t)*elementCount, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(start_event, 0);

    radx::sort(keysDev, keysDevBackup, elementCount);

    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&totalTime, start_event, stop_event);
    cudaDeviceSynchronize();

    cudaMemcpy(sortedNumbers.data(), keysDev, sizeof(uint32_t)*elementCount, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // copy from device to host (finally)
    std::cout << "Thrust sort measured in " << double(totalTime) << "ms" << std::endl;
    //std::cout << "Thrust sort measured in " << (double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6) << "ms" << std::endl;

    return 0;
};
