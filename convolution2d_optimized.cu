#include <iomanip>  // For formatted printing
#include <cstdlib>
#include <cmath>    // For fabs function
#include <iostream>
#include <sys/time.h>
#include <math.h>
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaMalloc             hipMalloc 
#define cudaFree               hipFree
#define cudaHostMalloc         hipHostMalloc
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyToSymbol     hipMemcpyToSymbol
#define cudaMemset             hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaError_t            hipError_t
#else
#include <cuda.h>
#endif
#include <fstream>
#include <cstdlib>

#define TILE_WIDTH 16
#define KERNEL_RADIUS 1
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

// Adjust the size of the constant memory to accommodate your kernel size and channels
__constant__ float d_kernel_const[28 * 3 * KERNEL_SIZE * KERNEL_SIZE]; // [output_channels][input_channels][kernel_height][kernel_width]

// Optimized CUDA kernel for 2D convolution using shared memory and constant memory
__global__ void convolution2D_optimized(
    float* d_input,
    float* d_output,
    int input_height,
    int input_width,
    int input_channels,
    int output_channels
) {
    // Shared memory tile with halo cells
    extern __shared__ float s_input[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_WIDTH + ty;
    int col_o = blockIdx.x * TILE_WIDTH + tx;

    int row_i = row_o - KERNEL_RADIUS;
    int col_i = col_o - KERNEL_RADIUS;

    int shared_mem_width = TILE_WIDTH + 2 * KERNEL_RADIUS;

    // Calculate the number of elements per input channel in shared memory
    int shared_mem_per_channel = shared_mem_width * shared_mem_width;

    // Load input data into shared memory for each input channel
    for (int ic = 0; ic < input_channels; ++ic) {
        // Pointer to the start of shared memory for this channel
        float* s_channel = &s_input[ic * shared_mem_per_channel];

        // Load data into shared memory
        if ((row_i >= 0) && (row_i < input_height) && (col_i >= 0) && (col_i < input_width)) {
            int input_idx = ((ic * input_height + row_i) * input_width) + col_i;
            s_channel[ty * shared_mem_width + tx] = d_input[input_idx];
        } else {
            s_channel[ty * shared_mem_width + tx] = 0.0f; // Zero-padding for out-of-bounds
        }
    }

    __syncthreads();

    // Perform convolution if within output bounds
    if (ty < TILE_WIDTH && tx < TILE_WIDTH && row_o < input_height && col_o < input_width) {
        // Loop over output channels
        for (int oc = 0; oc < output_channels; ++oc) {
            float output_value = 0.0f;

            // Loop over input channels
            for (int ic = 0; ic < input_channels; ++ic) {
                // Pointer to the start of shared memory for this channel
                float* s_channel = &s_input[ic * shared_mem_per_channel];

                // Convolution operation
                // #pragma unroll
                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    // #pragma unroll
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        int s_row = ty + ky;
                        int s_col = tx + kx;
                        float input_value = s_channel[s_row * shared_mem_width + s_col];

                        int kernel_idx = (((oc * input_channels + ic) * KERNEL_SIZE + ky) * KERNEL_SIZE) + kx;
                        float kernel_value = d_kernel_const[kernel_idx];

                        output_value += input_value * kernel_value;
                    }
                }
            }

            int output_idx = ((oc * input_height + row_o) * input_width) + col_o;
            d_output[output_idx] = output_value;
        }
    }
}

double get_time_in_ms() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}

int main() {
    cudaError_t GE;

    // Input dimensions
    int input_height = 2048;
    int input_width = 2048;
    int input_channels = 3;

    // Kernel dimensions
    int kernel_height = KERNEL_SIZE;
    int kernel_width = KERNEL_SIZE;
    int output_channels = 28; // Number of output channels

    // Output dimensions
    int output_height = input_height;
    int output_width = input_width;

    // Allocate host memory
    size_t input_size = input_height * input_width * input_channels * sizeof(float);
    size_t kernel_size = kernel_height * kernel_width * input_channels * output_channels * sizeof(float);
    size_t output_size = output_height * output_width * output_channels * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size);
    float* h_output = (float*)malloc(output_size);

    // Initialize input and kernel with random values
    srand(0); // Seed for reproducibility
    for (int i = 0; i < input_height * input_width * input_channels; ++i) {
        h_input[i] = rand() % 10;
    }

    for (int i = 0; i < kernel_height * kernel_width * input_channels * output_channels; ++i) {
        h_kernel[i] = 1.0f; // Adjust as needed
    }

    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;

    GE = cudaMalloc((void**)&d_input, input_size);
    GE = cudaMalloc((void**)&d_output, output_size);

    // Copy data from host to device
    GE = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    GE = cudaMemcpyToSymbol(d_kernel_const, h_kernel, kernel_size);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH + 2 * KERNEL_RADIUS, TILE_WIDTH + 2 * KERNEL_RADIUS);
    dim3 dimGrid(
        (input_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (input_height + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Calculate shared memory size
    int shared_mem_size = input_channels * (TILE_WIDTH + 2 * KERNEL_RADIUS) * (TILE_WIDTH + 2 * KERNEL_RADIUS) * sizeof(float);
    double start_time = get_time_in_ms();
    // Launch the optimized convolution kernel
    convolution2D_optimized<<<dimGrid, dim3(TILE_WIDTH + 2 * KERNEL_RADIUS, TILE_WIDTH + 2 * KERNEL_RADIUS), shared_mem_size>>>(
        d_input, d_output,
        input_height, input_width, input_channels,
        output_channels
    );
    double end_time = get_time_in_ms();
    printf("Time required to complete conversation %f milliseconds\n", (end_time - start_time));

    // Copy the result back to host memory
    GE = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Save input, kernel, and output to files
    std::ofstream input_file("input2.bin", std::ios::out | std::ios::binary);
    std::ofstream kernel_file("kernel2.bin", std::ios::out | std::ios::binary);
    std::ofstream output_file("output2.bin", std::ios::out | std::ios::binary);

    if (!input_file || !kernel_file || !output_file) {
        std::cerr << "Error opening file for writing." << std::endl;
        // Free memory before exiting
        GE = cudaFree(d_input);
        GE = cudaFree(d_output);
        free(h_input);
        free(h_kernel);
        free(h_output);
        return -1;
    }

    // Write data to files
    input_file.write(reinterpret_cast<char*>(h_input), input_size);
    kernel_file.write(reinterpret_cast<char*>(h_kernel), kernel_size);
    output_file.write(reinterpret_cast<char*>(h_output), output_size);

    // Close files
    input_file.close();
    kernel_file.close();
    output_file.close();

    std::cout << "Input, kernel, and output have been saved to 'input.bin', 'kernel.bin', and 'output.bin' respectively." << std::endl;

    // Free device memory
    GE = cudaFree(d_input);
    GE = cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
