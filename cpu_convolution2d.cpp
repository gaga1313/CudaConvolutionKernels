#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Function to perform 2D convolution
void convolution2D(const std::vector<std::vector<std::vector<float>>>& input,
                   const std::vector<std::vector<std::vector<std::vector<float>>>>& kernel,
                   std::vector<std::vector<std::vector<float>>>& output) {
    int inputChannels = input.size();
    int inputHeight = input[0].size();
    int inputWidth = input[0][0].size();
    int kernelCount = kernel.size();
    int kernelChannels = kernel[0].size();
    int kernelHeight = kernel[0][0].size();
    int kernelWidth = kernel[0][0][0].size();
    int outputHeight = output[0].size();
    int outputWidth = output[0][0].size();

    for (int k = 0; k < kernelCount; ++k) {
        for (int i = 0; i < outputHeight; ++i) {
            for (int j = 0; j < outputWidth; ++j) {
                float sum = 0.0f;
                for (int c = 0; c < inputChannels; ++c) {
                    for (int m = 0; m < kernelHeight; ++m) {
                        for (int n = 0; n < kernelWidth; ++n) {
                            int inputI = i + m - kernelHeight / 2;
                            int inputJ = j + n - kernelWidth / 2;
                            if (inputI >= 0 && inputI < inputHeight && inputJ >= 0 && inputJ < inputWidth) {
                                sum += input[c][inputI][inputJ] * kernel[k][c][m][n];
                            }
                        }
                    }
                }
                output[k][i][j] = sum;
            }
        }
    }
}

// Function to initialize a matrix with random values
template<typename T>
void initializeRandom(std::vector<T>& matrix) {
    for (auto& elem : matrix) {
        initializeRandom(elem);
    }
}

template<>
void initializeRandom(std::vector<float>& matrix) {
    for (auto& elem : matrix) {
        elem = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    // Initialize input matrix (3 x 32 x 32)
    std::vector<std::vector<std::vector<float>>> input(3, std::vector<std::vector<float>>(32, std::vector<float>(32)));
    initializeRandom(input);

    // Initialize kernel (28 x 3 x 3 x 3)
    std::vector<std::vector<std::vector<std::vector<float>>>> kernel(28, std::vector<std::vector<std::vector<float>>>(3, std::vector<std::vector<float>>(3, std::vector<float>(3))));
    initializeRandom(kernel);

    // Initialize output matrix (28 x 32 x 32)
    std::vector<std::vector<std::vector<float>>> output(28, std::vector<std::vector<float>>(32, std::vector<float>(32)));

    // Perform 2D convolution and measure time
    auto start = std::chrono::high_resolution_clock::now();
    convolution2D(input, kernel, output);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print the time taken
    std::cout << "Time taken for convolution: " << duration.count() << " milliseconds" << std::endl;

    // Print a sample of the output (first 5x5 of the first channel)
    std::cout << "\nSample output (5x5 of first channel):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << output[0][i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}