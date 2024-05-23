
import numpy as np

def convolve2d(input_array, kernel_array, mode='full'):
    """
    Perform 2D convolution between input_array and kernel_array.
    """
    # Get dimensions of input_array and kernel_array
    input_height, input_width = input_array.shape
    kernel_height, kernel_width = kernel_array.shape

    # Calculate output dimensions based on mode
    if mode == 'full':
        output_height = input_height + kernel_height - 1
        output_width = input_width + kernel_width - 1
        padding_height = kernel_height - 1
        padding_width = kernel_width - 1
    elif mode == 'valid':
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        padding_height = 0
        padding_width = 0
    else:
        raise ValueError("Invalid mode. Mode must be either 'full' or 'valid'.")

    # Pad input_array
    padded_input = np.pad(input_array, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    # Initialize result array
    result = np.zeros((output_height, output_width))
    kernel_array = np.flipud(np.fliplr(kernel_array))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            input_slice = padded_input[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(input_slice * kernel_array)

    return result

def correlate2d(input_array, kernel_array, mode='valid'):
    """
    Perform 2D correlation between input_array and kernel_array.
    """
    # Get dimensions of input_array and kernel_array
    input_height, input_width = input_array.shape
    kernel_height, kernel_width = kernel_array.shape

    # Calculate output dimensions based on mode
    if mode == 'valid':
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1
        padding_height = 0
        padding_width = 0
    elif mode == 'full':
        output_height = input_height + kernel_height - 1
        output_width = input_width + kernel_width - 1
        padding_height = kernel_height - 1
        padding_width = kernel_width - 1
    else:
        raise ValueError("Invalid mode. Mode must be either 'valid' or 'full'.")

    padded_input = np.pad(input_array, ((padding_height, padding_height), (padding_width, padding_width)),
                          mode='constant')

    # Initialize result array
    result = np.zeros((output_height, output_width))

    # Perform correlation
    for i in range(output_height):
        for j in range(output_width):
            input_slice = padded_input[i:i + kernel_height, j:j + kernel_width]
            if input_slice.shape == kernel_array.shape:
                result[i, j] = np.sum(input_slice * kernel_array)

    return result


class ConvolutionalLayer:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

