#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <fstream>
#include <cstring>
#include <mpi.h>
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

// Kernel in constant memory
__constant__ float d_kernel_const[28 * 3 * KERNEL_SIZE * KERNEL_SIZE];

static inline double get_time_in_ms() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1000.0 + time.tv_usec / 1000.0;
}

// Kernel for local convolution with halo
__global__ void convolution2D_local(
    float* d_input,
    float* d_output,
    int extended_height,
    int input_width,
    int local_height,
    int input_channels,
    int output_channels,
    int halo
) {
    extern __shared__ float s_input[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int shared_mem_width = TILE_WIDTH + 2 * KERNEL_RADIUS;
    int shared_mem_per_channel = shared_mem_width * shared_mem_width;

    int row_o = blockIdx.y * TILE_WIDTH + ty; // 0 ... extended_height-1
    int col_o = blockIdx.x * TILE_WIDTH + tx; // 0 ... input_width-1

    // Load from d_input to shared mem
    for (int ic = 0; ic < input_channels; ic++) {
        float* s_channel = &s_input[ic * shared_mem_per_channel];
        float val = 0.0f;
        if (row_o < extended_height && col_o < input_width) {
            int input_idx = (row_o * input_width * input_channels) + (col_o * input_channels) + ic;
            val = d_input[input_idx];
        }
        s_channel[ty * shared_mem_width + tx] = val;
    }

    __syncthreads();

    // Only process interior region: [halo..halo+local_height-1, full width]
    if (row_o >= halo && row_o < halo + local_height && col_o < input_width) {
        for (int oc = 0; oc < output_channels; oc++) {
            float output_value = 0.0f;
            for (int ic = 0; ic < input_channels; ic++) {
                float* s_channel = &s_input[ic * shared_mem_per_channel];
                #pragma unroll
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    #pragma unroll
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int s_row = ty + ky;
                        int s_col = tx + kx;
                        float input_val = s_channel[s_row * shared_mem_width + s_col];

                        int kernel_idx = (((oc * input_channels + ic) * KERNEL_SIZE + ky) * KERNEL_SIZE) + kx;
                        float kernel_val = d_kernel_const[kernel_idx];

                        output_value += input_val * kernel_val;
                    }
                }
            }

            int out_r = row_o - halo; // map back to local coords [0..local_height-1]
            int out_c = col_o;
            int output_idx = (out_r * input_width * output_channels) + (out_c * output_channels) + oc;
            d_output[output_idx] = output_value;
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input dimensions (global)
    int input_height = 8192; // adjust as needed (large image)
    int input_width = 8192;
    int input_channels = 3;
    int output_channels = 28;

    // Check divisibility
    if (input_height % world_size != 0) {
        if (rank == 0) std::cerr << "input_height must be divisible by world_size.\n";
        MPI_Finalize();
        return -1;
    }

    int local_height = input_height / world_size;
    int halo = KERNEL_RADIUS;

    // Kernel dimensions
    int kernel_height = KERNEL_SIZE;
    int kernel_width = KERNEL_SIZE;

    size_t input_size_global = input_height * input_width * input_channels * sizeof(float);
    size_t kernel_size = kernel_height * kernel_width * input_channels * output_channels * sizeof(float);
    size_t output_size_global = input_height * input_width * output_channels * sizeof(float);

    // Extended domain
    int extended_height = local_height + 2 * halo;
    size_t local_input_size = extended_height * input_width * input_channels * sizeof(float);
    size_t local_output_size = local_height * input_width * output_channels * sizeof(float);

    float* h_input = nullptr;
    float* h_kernel = nullptr;
    float* h_output = nullptr;

    if (rank == 0) {
        h_input = (float*)malloc(input_size_global);
        h_kernel = (float*)malloc(kernel_size);
        h_output = (float*)malloc(output_size_global);

        srand(0);
        // Fill input in row-major with channels last:
        for (int r = 0; r < input_height; r++) {
            for (int c = 0; c < input_width; c++) {
                for (int ic = 0; ic < input_channels; ic++) {
                    int idx = (r * input_width * input_channels) + (c * input_channels) + ic;
                    h_input[idx] = (float)(rand() % 10);
                }
            }
        }

        for (int i = 0; i < kernel_height * kernel_width * input_channels * output_channels; i++) {
            h_kernel[i] = 1.0f;
        }
    } else {
        h_kernel = (float*)malloc(kernel_size);
    }

    // Broadcast kernel
    MPI_Bcast(h_kernel, (int)(kernel_size / sizeof(float)), MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Scatter the input rows
    float* local_input_nohalo = (float*)malloc(local_height * input_width * input_channels * sizeof(float));
    // Each rank gets a continuous block of rows:
    // offset in floats = rank * local_height * input_width * input_channels
    MPI_Scatter(h_input, (int)((local_height * input_width * input_channels)), MPI_FLOAT,
                local_input_nohalo, (int)((local_height * input_width * input_channels)), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    // Assign distinct GPU per rank
    int num_devices;
    cudaError_t GE = cudaGetDeviceCount(&num_devices);
    if (num_devices < world_size) {
        if (rank == 0) std::cerr << "Not enough GPUs.\n";
        MPI_Finalize();
        return -1;
    }
    GE = cudaSetDevice(rank);
    if (GE != cudaSuccess) {
        std::cerr << "Rank " << rank << " failed to set GPU device.\n";
        MPI_Finalize();
        return -1;
    }

    float* h_input_extended = (float*)malloc(local_input_size);
    memset(h_input_extended, 0, local_input_size);

    // Place local data in center of extended array
    for (int ic = 0; ic < input_channels; ic++) {
        for (int r = 0; r < local_height; r++) {
            memcpy(&h_input_extended[(ic * extended_height + (r + halo)) * input_width],
                   &local_input_nohalo[(ic * local_height + r) * input_width],
                   input_width * sizeof(float));
        }
    }
    free(local_input_nohalo);

    // Determine neighbors for vertical halo exchange
    int north = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int south = (rank < world_size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Halo buffers
    int row_halo_size = input_width * input_channels;
    float* top_send = (float*)malloc(row_halo_size * sizeof(float));
    float* top_recv = (float*)malloc(row_halo_size * sizeof(float));
    float* bottom_send = (float*)malloc(row_halo_size * sizeof(float));
    float* bottom_recv = (float*)malloc(row_halo_size * sizeof(float));

    // Pack top halo
    if (north != MPI_PROC_NULL) {
        int top_row = halo; // first interior row
        for (int ic = 0; ic < input_channels; ic++) {
            memcpy(&top_send[ic * input_width],
                   &h_input_extended[(ic * extended_height + top_row) * input_width],
                   input_width * sizeof(float));
        }
    }

    // Pack bottom halo
    if (south != MPI_PROC_NULL) {
        int bottom_row = halo + local_height - 1;
        for (int ic = 0; ic < input_channels; ic++) {
            memcpy(&bottom_send[ic * input_width],
                   &h_input_extended[(ic * extended_height + bottom_row) * input_width],
                   input_width * sizeof(float));
        }
    }

    MPI_Request requests[4];
    int req_count = 0;

    // Halo exchange
    if (north != MPI_PROC_NULL) {
        MPI_Irecv(top_recv, row_halo_size, MPI_FLOAT, north, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(top_send, row_halo_size, MPI_FLOAT, north, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (south != MPI_PROC_NULL) {
        MPI_Irecv(bottom_recv, row_halo_size, MPI_FLOAT, south, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Isend(bottom_send, row_halo_size, MPI_FLOAT, south, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Place received halos
    if (north != MPI_PROC_NULL) {
        int top_row = halo - 1;
        for (int ic = 0; ic < input_channels; ic++) {
            memcpy(&h_input_extended[(ic * extended_height + top_row) * input_width],
                   &top_recv[ic * input_width],
                   input_width * sizeof(float));
        }
    }

    if (south != MPI_PROC_NULL) {
        int bottom_row = halo + local_height;
        for (int ic = 0; ic < input_channels; ic++) {
            memcpy(&h_input_extended[(ic * extended_height + bottom_row) * input_width],
                   &bottom_recv[ic * input_width],
                   input_width * sizeof(float));
        }
    }

    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    GE = cudaMalloc((void**)&d_input, local_input_size);
    GE = cudaMalloc((void**)&d_output, local_output_size);
    GE = cudaMemcpy(d_input, h_input_extended, local_input_size, cudaMemcpyHostToDevice);

    // Copy kernel to device
    GE = cudaMemcpyToSymbol(d_kernel_const, h_kernel, kernel_size);

    dim3 dimBlock(TILE_WIDTH + 2 * KERNEL_RADIUS, TILE_WIDTH + 2 * KERNEL_RADIUS);
    dim3 dimGrid(
        (input_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (extended_height + TILE_WIDTH - 1) / TILE_WIDTH
    );

    int shared_mem_size = input_channels * (TILE_WIDTH + 2 * KERNEL_RADIUS) * (TILE_WIDTH + 2 * KERNEL_RADIUS) * sizeof(float);

    double start_time = get_time_in_ms();
    convolution2D_local<<<dimGrid, dimBlock, shared_mem_size>>>(
        d_input, d_output,
        extended_height, input_width, local_height,
        input_channels, output_channels, halo
    );
    GE = cudaDeviceSynchronize();
    double end_time = get_time_in_ms();
    double local_time = end_time - start_time;

    // Copy local output
    float* local_output = (float*)malloc(local_output_size);
    GE = cudaMemcpy(local_output, d_output, local_output_size, cudaMemcpyDeviceToHost);

    // Gather all outputs at rank 0
    float* gather_buffer = nullptr;
    if (rank == 0) {
        gather_buffer = (float*)malloc(local_output_size * world_size);
    }

    MPI_Gather(local_output, (int)(local_output_size/sizeof(float)), MPI_FLOAT,
               gather_buffer, (int)(local_output_size/sizeof(float)), MPI_FLOAT,
               0, MPI_COMM_WORLD);

    // Collect timing info
    double max_time = local_time;
    if (rank == 0) {
        for (int r = 1; r < world_size; r++) {
            double other_time;
            MPI_Recv(&other_time, 1, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (other_time > max_time) max_time = other_time;
        }
        std::cout << "Max time required: " << max_time << " ms\n";

        // Reconstruct global output from gather_buffer:
        // Each rank has local_height rows, full width, output_channels
        // rank r = rows [r*local_height ... (r+1)*local_height-1]
        for (int r = 0; r < world_size; r++) {
            int start_row = r * local_height;
            for (int oc = 0; oc < output_channels; oc++) {
                for (int rr = 0; rr < local_height; rr++) {
                    memcpy(&h_output[((start_row + rr) * input_width * output_channels) + oc],
                           &gather_buffer[(r * (local_output_size/sizeof(float))) + (rr * input_width * output_channels) + oc],
                           sizeof(float) * (input_width - 0) ); // full width
                }
            }
        }

        // Save data
        std::ofstream input_file("input2.bin", std::ios::out | std::ios::binary);
        std::ofstream kernel_file("kernel2.bin", std::ios::out | std::ios::binary);
        std::ofstream output_file("output2.bin", std::ios::out | std::ios::binary);

        if (input_file && kernel_file && output_file) {
            input_file.write(reinterpret_cast<char*>(h_input), input_size_global);
            kernel_file.write(reinterpret_cast<char*>(h_kernel), kernel_size);
            output_file.write(reinterpret_cast<char*>(h_output), output_size_global);
            input_file.close();
            kernel_file.close();
            output_file.close();
            std::cout << "Input, kernel, and output saved to 'input2.bin', 'kernel2.bin', 'output2.bin'.\n";
        } else {
            std::cerr << "Error opening files for writing.\n";
        }

        free(h_input);
        free(h_output);
        free(gather_buffer);
    } else {
        MPI_Send(&local_time, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    free(h_kernel);
    free(h_input_extended);
    free(local_output);

    free(top_send); free(top_recv);
    free(bottom_send); free(bottom_recv);

    GE = cudaFree(d_input);
    GE = cudaFree(d_output);

    MPI_Finalize();
    return 0;
}
