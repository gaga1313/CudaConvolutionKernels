import numpy as np
import cv2

# Define the dimensions (must match those in the C++ code)
input_height = 32
input_width = 32
input_channels = 3

kernel_height = 3
kernel_width = 3
kernel_channels = 28

output_height = input_height
output_width = input_width

# Read input tensor
input_size = input_height * input_width * input_channels
input_data = np.fromfile('input2Dconv.bin', dtype=np.float32, count=input_size)
input_data = input_data.reshape((input_channels, input_height, input_width))

# Read kernel tensor
kernel_size = kernel_height * kernel_width * input_channels * kernel_channels
kernel_data = np.fromfile('kernel2Dconv.bin', dtype=np.float32, count=kernel_size)
kernel_data = kernel_data.reshape((kernel_channels, input_channels, kernel_height, kernel_width))

# Read output tensor from CUDA
output_size = output_height * output_width * kernel_channels
output_data_cuda = np.fromfile('output2Dconv.bin', dtype=np.float32, count=output_size)
output_data_cuda = output_data_cuda.reshape((kernel_channels, output_height, output_width))

# import ipdb; ipdb.set_trace()
# Perform convolution in Python using OpenCV
output_data_python = np.zeros_like(output_data_cuda)

for k in range(kernel_channels):
    output = np.zeros((output_height, output_width), dtype=np.float32)
    for c in range(input_channels):
        # Input image for this channel
        input_channel = input_data[c]
        # Kernel for this output channel and input channel
        kernel = kernel_data[k, c]
        # Perform convolution using cv2.filter2D
        conv_result = cv2.filter2D(input_channel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        output += conv_result
    output_data_python[k] = output

# Compare outputs
difference = np.abs(output_data_cuda - output_data_python)
max_difference = np.max(difference)
print(f"Maximum difference between CUDA and OpenCV outputs: {max_difference}")

if max_difference < 1e-4:
    print("The outputs match closely!")
else:
    print("There are significant differences between the outputs.")

# Optionally, print indices where differences occur
threshold = 1e-4
indices = np.where(difference > threshold)
if indices[0].size > 0:
    print(f"Differences found at indices: {indices}")
else:
    print("No significant differences found.")
