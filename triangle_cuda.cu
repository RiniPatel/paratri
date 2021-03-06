#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "cycletimer.h"

#define MAX_TRIANGLES 20000000
#define MAX_VERTICE 4000000

#define LOG2_THREADS_PER_BLOCK 9
#define THREADS_PER_BLOCK (1U << LOG2_THREADS_PER_BLOCK)

#define MAX_SUBBLOCK (10000/THREADS_PER_BLOCK)

/* Helper function to round up to a power of 2. 
 */
inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void
aggregate_kernel(uint32_t N, uint32_t *device_count, uint32_t *triangle_list, uint32_t *device_output)
{
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int idx = (blockIdx.x << LOG2_THREADS_PER_BLOCK) + threadIdx.x;
    
    uint32_t start = (idx == 0)? 0 : device_count[idx - 1];
    uint32_t end = device_count[idx];
    
    // call cuda kernel
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int verticePerBlock = THREADS_PER_BLOCK;
    const int blocks = (N + verticePerBlock - 1) / verticePerBlock;
    const int totalThreads = (threadsPerBlock * blocks);
    int thr_idx = idx * (MAX_TRIANGLES/totalThreads);

    for(int c = start; c < end; c++) {
        int oidx = 3 * c;
        int iidx = 3 * (c - start);
        device_output[oidx] = triangle_list[thr_idx + iidx];
        device_output[oidx + 1] = triangle_list[thr_idx + iidx + 1];
        device_output[oidx + 2] = triangle_list[thr_idx + iidx + 2];        
    }
}

__global__ void
child(uint32_t *IA, uint32_t *JA, int num_nnz_y, int totalThreads, uint32_t *x_col_begin, uint32_t *x_col_end, uint32_t *sub_delta)
{
    __shared__ uint32_t sharedTriangleCount[THREADS_PER_BLOCK];
    int j = (blockIdx.x << LOG2_THREADS_PER_BLOCK) + threadIdx.x;
    int delta = 0;
    uint32_t num_nnz_x = (x_col_end - x_col_begin);

    if (j < num_nnz_y) {
        uint32_t *x_col = x_col_begin;
        uint32_t *A_col = JA + IA[x_col_end[j]];
        uint32_t *A_col_max = JA + IA[x_col_end[j] + 1];

        // this loop searches through all possible vertices for z.
        for (uint32_t k = 0; k < num_nnz_x; ++k) {
            while ((*A_col < x_col[k])  && (A_col < A_col_max)) ++A_col;

            // for triangle enumeration i, *x_col_end, *A_col
            if (*A_col == x_col[k]) {
                // int idx = delta * 3;
                // uint32_t* ptr = (uint32_t*)(triangle_list + globalThreadIdx * (MAX_TRIANGLES/totalThreads));
                // ptr[idx] = i;
                // ptr[idx+1] = *A_col;
                // ptr[idx+2] = x_col_end[j];                
                delta++;
            }
        }        
    }
    sharedTriangleCount[threadIdx.x] = delta;

    __syncthreads();

    if (threadIdx.x == 0)
    {
        // disable CDP here by using seq thrust
        sub_delta[blockIdx.x] = thrust::reduce(thrust::seq, sharedTriangleCount, sharedTriangleCount + THREADS_PER_BLOCK);
    }
}

__global__ void
triangle_kernel(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t *device_count, uint32_t *triangle_list, uint32_t *sub_delta)
{
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int globalThreadIdx = (blockIdx.x << LOG2_THREADS_PER_BLOCK) + threadIdx.x;

    if (globalThreadIdx > N - 2) {
        return;
    }

    int i = globalThreadIdx + 1;

    // call cuda kernel
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int totalThreads = (threadsPerBlock * blocks);

    int delta = 0;

    uint32_t num_nnz_curr_row_x = IA[i + 1] - IA[i];
    uint32_t *x_col_begin = &JA[IA[i]];
    uint32_t *row_bound = &JA[IA[i + 1]];
    uint32_t *x_col_end = row_bound;


    for (uint32_t idx = 0; idx < num_nnz_curr_row_x; idx++)
    {
        if (x_col_begin[idx] > (i - 1)) {
            x_col_end = &x_col_begin[idx];
            break;
        }
    }

    uint32_t num_nnz_y = (row_bound - x_col_end);
    uint32_t num_nnz_x = (x_col_end - x_col_begin);
    const int subBlocks = (num_nnz_y + threadsPerBlock - 1) / threadsPerBlock;

    if (num_nnz_y > 64) {
        sub_delta = sub_delta + (MAX_SUBBLOCK * globalThreadIdx);
        child<<<subBlocks, threadsPerBlock>>>(IA,  JA, num_nnz_y, totalThreads, x_col_begin, x_col_end, sub_delta);
        cudaDeviceSynchronize();
        thrust::device_ptr<uint32_t> d_ptr = thrust::device_pointer_cast(sub_delta);
        delta = thrust::reduce(thrust::device, d_ptr, d_ptr + subBlocks);
    }
    else {
        // This is where the real triangle counting begins.
        // We search through all possible vertices for x
        for (uint32_t j = 0; j < num_nnz_y; ++j) {
            uint32_t *x_col = x_col_begin;
            uint32_t *A_col = JA + IA[x_col_end[j]];
            uint32_t *A_col_max = JA + IA[x_col_end[j] + 1];

            // this loop searches through all possible vertices for z.
            for (uint32_t k = 0; k < num_nnz_x; ++k) {
                while ((*A_col < x_col[k])  && (A_col < A_col_max)) ++A_col;

                // for triangle enumeration i, *x_col_end, *A_col
                if (*A_col == x_col[k]) {
                    int idx = delta * 3;
                    uint32_t* ptr = (uint32_t*)(triangle_list + globalThreadIdx * (MAX_TRIANGLES/totalThreads));
                    ptr[idx] = i;
                    ptr[idx+1] = *A_col;
                    ptr[idx+2] = x_col_end[j];                
                    delta++;
                }
            }
        }
    }

    device_count[globalThreadIdx] = delta;
}

uint32_t count_triangles_cuda(uint32_t *IA, uint32_t *JA, uint32_t N, uint32_t NUM_A, uint32_t * output)
{
    uint32_t num_triangles = 0;
    
    uint32_t *device_IA, *device_JA;
    uint32_t *device_count, *triangle_list;
    uint32_t *device_output, *sub_delta;

    // call cuda kernel
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int totalThreads = (threadsPerBlock * blocks);

    cudaMalloc((void **)&device_IA, (N + 1) * sizeof(uint32_t));
    cudaMalloc((void **)&device_JA, NUM_A * sizeof(uint32_t));
    cudaMemcpy(device_IA, IA, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(device_JA, JA, NUM_A * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    cudaMalloc((void **)&triangle_list, MAX_TRIANGLES * sizeof(uint32_t) * 3);
    cudaMalloc((void **)&device_count, totalThreads * sizeof(uint32_t));
    cudaMalloc((void **)&sub_delta, MAX_VERTICE * MAX_SUBBLOCK * sizeof(uint32_t));
    
    cudaMalloc((void **)&device_output, MAX_TRIANGLES * sizeof(uint32_t) * 3);

    double start = currentSeconds();

    triangle_kernel<<<blocks, threadsPerBlock>>>(device_IA, device_JA, N, NUM_A, device_count, triangle_list, sub_delta);
    cudaThreadSynchronize();

    thrust::device_ptr<uint32_t> device_count_thrust = thrust::device_pointer_cast(device_count);
    thrust::inclusive_scan(device_count_thrust, device_count_thrust + totalThreads, device_count_thrust);

    aggregate_kernel<<<blocks, threadsPerBlock>>>(N, device_count, triangle_list, device_output);

    double end = currentSeconds();
    printf("CUDA computation time is %lf\n", end - start);

    cudaMemcpy(&num_triangles, &device_count[totalThreads - 1], sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(triangle_list, device_output, sizeof(uint32_t) * num_triangles * 3,
               cudaMemcpyDeviceToHost);

    cudaFree(device_IA);
    cudaFree(device_JA);
    cudaFree(device_count);
    cudaFree(device_output);
    cudaFree(triangle_list);
    cudaFree(sub_delta);

    return num_triangles;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
