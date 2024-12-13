import numpy as np
from scipy import ndimage

# Define the dimensions (must match those in the CUDA code)
input_depth = 32
input_height = 32
input_width = 32

kernel_depth = 3  # KERNEL_SIZE from the CUDA code
kernel_height = 3
kernel_width = 3

output_depth = input_depth
output_height = input_height
output_width = input_width

# Read input tensor
input_size = input_depth * input_height * input_width
input_data = np.fromfile('input3D.bin', dtype=np.float32, count=input_size)
input_data = input_data.reshape((input_depth, input_height, input_width))

# Read kernel tensor
kernel_size = kernel_depth * kernel_height * kernel_width
kernel_data = np.fromfile('kernel3D.bin', dtype=np.float32, count=kernel_size)
kernel_data = kernel_data.reshape((kernel_depth, kernel_height, kernel_width))

# Perform convolution in Python using SciPy
output_data_python = ndimage.convolve(input_data, kernel_data, mode='constant', cval=0.0)

# Read output tensor from CUDA
output_size = output_depth * output_height * output_width
output_data_cuda = np.fromfile('output3D.bin', dtype=np.float32, count=output_size)
output_data_cuda = output_data_cuda.reshape((output_depth, output_height, output_width))

import ipdb;ipdb.set_trace()
# Compare outputs
difference = np.abs(output_data_cuda - output_data_python)
max_difference = np.max(difference)
print(f"Maximum difference between CUDA and Python outputs: {max_difference}")

# Check if the outputs match closely
tolerance = 1e-4
if max_difference < tolerance:
    print("The outputs match closely!")
else:
    print("There are significant differences between the outputs.")

# Optionally, print indices where differences occur
indices = np.where(difference > tolerance)
if indices[0].size > 0:
    num_differences = indices[0].size
    print(f"Differences found at {num_differences} positions exceeding the tolerance of {tolerance}.")
else:
    print("No significant differences found.")
