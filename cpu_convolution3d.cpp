#include <iostream>
#include <vector>
#include <random>
#include <chrono>

constexpr int INPUT_SIZE = 32;
constexpr int KERNEL_SIZE = 3;
constexpr int OUTPUT_SIZE = 32;
constexpr int PAD = 1;

using Tensor3D = std::vector<std::vector<std::vector<float>>>;

Tensor3D convolve3D(const Tensor3D& input, const Tensor3D& kernel) {
    Tensor3D output(OUTPUT_SIZE, std::vector<std::vector<float>>(OUTPUT_SIZE, std::vector<float>(OUTPUT_SIZE, 0.0f)));

    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            for (int k = 0; k < OUTPUT_SIZE; ++k) {
                float sum = 0.0f;
                for (int di = 0; di < KERNEL_SIZE; ++di) {
                    for (int dj = 0; dj < KERNEL_SIZE; ++dj) {
                        for (int dk = 0; dk < KERNEL_SIZE; ++dk) {
                            int ii = i + di - PAD;
                            int jj = j + dj - PAD;
                            int kk = k + dk - PAD;
                            if (ii >= 0 && ii < INPUT_SIZE && jj >= 0 && jj < INPUT_SIZE && kk >= 0 && kk < INPUT_SIZE) {
                                sum += input[ii][jj][kk] * kernel[di][dj][dk];
                            }
                        }
                    }
                }
                output[i][j][k] = sum;
            }
        }
    }
    return output;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    Tensor3D input(INPUT_SIZE, std::vector<std::vector<float>>(INPUT_SIZE, std::vector<float>(INPUT_SIZE)));
    Tensor3D kernel(KERNEL_SIZE, std::vector<std::vector<float>>(KERNEL_SIZE, std::vector<float>(KERNEL_SIZE)));

    // Initialize input and kernel with random values
    for (auto& layer : input) {
        for (auto& row : layer) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }
    for (auto& layer : kernel) {
        for (auto& row : layer) {
            for (auto& val : row) {
                val = dis(gen);
            }
        }
    }

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    Tensor3D output = convolve3D(input, kernel);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Convolution execution time: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
