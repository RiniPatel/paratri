#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "cycletimer.h"

extern float toBW(int bytes, float sec);

#define LOG2_THREADS_PER_BLOCK 9
#define THREADS_PER_BLOCK (1U << LOG2_THREADS_PER_BLOCK)

/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
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
exclusive_scan_upsweep_kernel(int * device_result, int length, int twod, int N)
{
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = (blockIdx.x << LOG2_THREADS_PER_BLOCK) + threadIdx.x;

    // shift by 1 to make twod1 = twod * 2
    int twod1 = twod << 1;

    // index should be multiple of twod1
    // twod1 is power of 2
    // val % power(2, n) = val & (power(2, n) - 1)
    if ((index & (twod1 - 1)) == 0 && index + twod1 - 1 < N) {
        device_result[index + twod1 - 1] += device_result[index + twod - 1];
    }
}

__global__ void
exclusive_scan_downsweep_kernel(int * device_result, int length, int twod, 
    int N)
{
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = (blockIdx.x << LOG2_THREADS_PER_BLOCK) + threadIdx.x;

    // shift by 1 to make twod1 = twod * 2
    int twod1 = twod << 1;

    // index should be multiple of twod1
    // twod1 is power of 2
    // val % power(2, n) = val & (power(2, n) - 1)
    if ((index & (twod1 - 1)) == 0 && index + twod1 - 1 < N) {
        int tmp = device_result[index + twod-1];
        device_result[index + twod-1] = device_result[index + twod1-1];
        device_result[index + twod1-1] += tmp;
    }
}

__global__ void
set_zero_kernel(int* device_result, int N)
{
    device_result[N-1] = 0;
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
   // rounded length to next power of 2 
    int N = nextPow2(length);

    // compute number of blocks and threads per block
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // upsweep phase
    for (int twod = 1; twod < N; twod*=2)
    {
        // run kernel
        exclusive_scan_upsweep_kernel<<<blocks, threadsPerBlock>>>
                                            (device_result, length, twod, N);
    }

    // device_result[N - 1] = 0
    set_zero_kernel<<<1,1>>>(device_result, N);
    
    for (int twod = N/2; twod >= 1; twod /= 2)
    {
        // run kernel
        exclusive_scan_downsweep_kernel<<<blocks, threadsPerBlock>>>
                                            (device_result, length, twod, N);
    }

}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void
fill_bool_array_kernel(int *device_input, int rounded_length, int * device_bool)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < rounded_length)
    {
        if (index != rounded_length && 
                device_input[index] == device_input[index + 1])
            device_bool[index] = 1;
        else
            device_bool[index] = 0;
    }
}

__global__ void
fill_repeat_index_kernel(int * device_input, int * device_bool, 
                         int rounded_length, int * device_output)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < rounded_length)
    {
        if (index != rounded_length && 
                device_input[index] == device_input[index + 1])
            device_output[device_bool[index]] = index;
    }
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */
    const int threadsPerBlock = THREADS_PER_BLOCK;
    const int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;

    int rounded_length = nextPow2(length);
    int * device_bool;
    cudaMalloc((void **) &device_bool, rounded_length * sizeof(int));

    fill_bool_array_kernel<<<blocks, threadsPerBlock>>>(device_input, 
        rounded_length, device_bool);

    exclusive_scan(device_bool, rounded_length, device_bool);


    fill_repeat_index_kernel<<<blocks, threadsPerBlock>>>(device_input, 
        device_bool, rounded_length, device_output);

    int num_elements;
    cudaMemcpy(&num_elements, &device_bool[length - 1], sizeof(int),
               cudaMemcpyDeviceToHost);


    return num_elements;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
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
