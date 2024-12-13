#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <math.h>
#include <sys/time.h>
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

#define TILE_WIDTH 4
#define KERNEL_RADIUS 1
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)

// Adjust the size of the constant memory to accommodate your kernel size
__constant__ float d_kernel_const[KERNEL_SIZE * KERNEL_SIZE * KERNEL_SIZE]; // [kernel_depth][kernel_height][kernel_width]

// Optimized CUDA kernel for 3D convolution using shared memory and constant memory
__global__ void convolution3D_optimized(
    float* d_input,
    float* d_output,
    int input_depth,
    int input_height,
    int input_width
) {
    // Calculate indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int output_x = blockIdx.x * TILE_WIDTH + tx;
    int output_y = blockIdx.y * TILE_WIDTH + ty;
    int output_z = blockIdx.z * TILE_WIDTH + tz;

    int input_x = output_x - KERNEL_RADIUS;
    int input_y = output_y - KERNEL_RADIUS;
    int input_z = output_z - KERNEL_RADIUS;

    // Shared memory tile with halo cells
    __shared__ float s_data[TILE_WIDTH + 2 * KERNEL_RADIUS][TILE_WIDTH + 2 * KERNEL_RADIUS][TILE_WIDTH + 2 * KERNEL_RADIUS];

    // Load data into shared memory
    if (input_x >= 0 && input_x < input_width &&
        input_y >= 0 && input_y < input_height &&
        input_z >= 0 && input_z < input_depth) {
        s_data[tz][ty][tx] = d_input[(input_z * input_height * input_width) + (input_y * input_width) + input_x];
    } else {
        s_data[tz][ty][tx] = 0.0f; // Zero-padding for out-of-bounds
    }

    __syncthreads();

    // Perform convolution if within output bounds
    if (tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH &&
        output_x < input_width && output_y < input_height && output_z < input_depth) {
        float output_value = 0.0f;

        // #pragma unroll
        for (int kz = 0; kz < KERNEL_SIZE; ++kz) {
            // #pragma unroll
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                // #pragma unroll
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    float input_value = s_data[tz + kz][ty + ky][tx + kx];
                    float kernel_value = d_kernel_const[(kz * KERNEL_SIZE * KERNEL_SIZE) + (ky * KERNEL_SIZE) + kx];
                    output_value += input_value * kernel_value;
                }
            }
        }

        d_output[(output_z * input_height * input_width) + (output_y * input_width) + output_x] = output_value;
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
    int input_depth = 32;
    int input_height = 32;
    int input_width = 32;

    // Kernel dimensions
    int kernel_depth = KERNEL_SIZE;
    int kernel_height = KERNEL_SIZE;
    int kernel_width = KERNEL_SIZE;

    // Output dimensions (same as input for 'valid' convolution)
    int output_depth = input_depth;
    int output_height = input_height;
    int output_width = input_width;

    // Allocate host memory
    size_t input_size = input_depth * input_height * input_width * sizeof(float);
    size_t kernel_size = kernel_depth * kernel_height * kernel_width * sizeof(float);
    size_t output_size = output_depth * output_height * output_width * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size);
    float* h_output = (float*)malloc(output_size);

    // Initialize input and kernel with random values
    srand(0); // Seed for reproducibility
    for (int i = 0; i < input_depth * input_height * input_width; ++i) {
        h_input[i] = static_cast<float>(rand() % 10);
    }

    for (int i = 0; i < kernel_depth * kernel_height * kernel_width; ++i) {
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
    dim3 dimBlock(TILE_WIDTH + 2 * KERNEL_RADIUS, TILE_WIDTH + 2 * KERNEL_RADIUS, TILE_WIDTH + 2 * KERNEL_RADIUS);
    dim3 dimGrid(
        (input_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (input_height + TILE_WIDTH - 1) / TILE_WIDTH,
        (input_depth + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Calculate shared memory size
    size_t shared_mem_size = (TILE_WIDTH + 2 * KERNEL_RADIUS) * (TILE_WIDTH + 2 * KERNEL_RADIUS) * (TILE_WIDTH + 2 * KERNEL_RADIUS) * sizeof(float);
    double start_time = get_time_in_ms();
    // Launch the optimized convolution kernel
    convolution3D_optimized<<<dimGrid, dimBlock, shared_mem_size>>>(
        d_input, d_output,
        input_depth, input_height, input_width
    );
    double end_time = get_time_in_ms();
    printf("Time required to complete conversation %f milliseconds\n", (end_time - start_time));

    // Copy the result back to host memory
    GE = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Save input, kernel, and output to files
    std::ofstream input_file("input3D.bin", std::ios::out | std::ios::binary);
    std::ofstream kernel_file("kernel3D.bin", std::ios::out | std::ios::binary);
    std::ofstream output_file("output3D.bin", std::ios::out | std::ios::binary);

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

    std::cout << "Input, kernel, and output have been saved to 'input3D.bin', 'kernel3D.bin', and 'output3D.bin' respectively." << std::endl;

    // Free device memory
    GE = cudaFree(d_input);
    GE = cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_kernel);
    free(h_output);

    return 0;
}
