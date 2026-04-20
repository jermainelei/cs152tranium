import numpy as np
import torch


"""
Performs a 2D convolution operation using PyTorch's built-in functionality.
Args:
    X (array-like): Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W (array-like): Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias (array-like): Bias tensor of shape (out_channels).
Returns:
    torch.Tensor: The result of the 2D convolution operation, with shape 
                  (batch_size, out_channels, output_height, output_width).
Note:
    This function uses `torch.nn.functional.conv2d` for the convolution operation.
    For more details, refer to the PyTorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
"""
def conv2d_torch(X, W, bias):
    X = torch.tensor(X)
    W = torch.tensor(W)
    bias = torch.tensor(bias)

    conv_out = torch.nn.functional.conv2d(X, W, bias)

    return conv_out
 

"""
Performs a 2D convolution operation using a naive NumPy-based implementation.
Args:
    X (array-like): Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W (array-like): Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias (array-like): Bias tensor of shape (out_channels).
Returns:
    np.ndarray: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    This is a naive implementation of 2D convolution using nested loops and basic NumPy operations.
    It is not optimized for performance and is intended for reference purposes to illustrate
    the underlying calculations, memory accesses, and looping structure of the convolution operation.
"""
def conv2d_numpy(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape

    H_out = 1 + (input_height - filter_height)
    W_out = 1 + (input_width - filter_width)

    out = np.zeros((batch_size, out_channels, H_out, W_out))
    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    x_ij = X[b, :, i : i + filter_height, j : j + filter_width]
                    out[b, c, i, j] = np.sum(x_ij * W[c]) + bias[c]

    return out


"""
Performs a 2D convolution operation using a NumPy-based implementation mapped to matrix multiplications.
Args:
    X (array-like): Input tensor of shape (batch_size, in_channels, input_height, input_width).
    W (array-like): Weight tensor of shape (out_channels, in_channels, filter_height, filter_width).
    bias (array-like): Bias tensor of shape (out_channels).
Returns:
    np.ndarray: The result of the 2D convolution operation, with shape 
                (batch_size, out_channels, output_height, output_width).
Note:
    This version mirrors a potential mapping to a NKI kernel, with tiling and matmul operations.
"""
def conv2d_numpy_nki(X, W, bias):
    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    # Initialize output array
    X_out = np.zeros((batch_size, out_channels, out_height, out_width), dtype=X.dtype)

    # Tiling dimensions
    c_in_tile = 128   # Matches nl.tile_size.pmax
    c_out_tile = 128  # Matches nl.tile_size.gemm_stationary_fmax
    n_tiles_c_in = in_channels // c_in_tile
    n_tiles_c_out = out_channels // c_out_tile

    # Load in the weights, arranged for matmuls
    # Shape: (c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width)
    w = np.zeros((c_in_tile, c_out_tile, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width), dtype=W.dtype)
    for c_out_tile_idx in range(n_tiles_c_out):
        for c_in_tile_idx in range(n_tiles_c_in):
            for i in range(filter_height):
                for j in range(filter_width):
                    # Load a weight tile of shape [c_out_tile, c_in_tile]
                    w_tile = W[c_out_tile_idx * c_out_tile: (c_out_tile_idx + 1) * c_out_tile,
                               c_in_tile_idx * c_in_tile: (c_in_tile_idx + 1) * c_in_tile, i, j]
                    # Transpose so that IC is the first dimension: [c_in_tile, c_out_tile] <- [c_out_tile, c_in_tile]
                    w[:, :, c_out_tile_idx, c_in_tile_idx, i, j] = w_tile.T

    # Process the images one-by-one
    for img in range(batch_size):
        # Process each output channel tile
        for c_out_tile_idx in range(n_tiles_c_out):
            # Convolve: for each output row, convolve over the input channel tiles and filter positions
            for out_row in range(out_height):
                # Accumulator for output row
                row_out = np.zeros((c_out_tile, out_width), dtype=np.float32)
                # Loop over the input channel tiles and filter positions, accumulating the output row
                for c_in_tile_idx in range(n_tiles_c_in):
                    for i in range(filter_height):
                        for j in range(filter_width):
                            # Select the weight tile for the current input and output channel tiles idx and filter position
                            w_tile = w[:, :, c_out_tile_idx, c_in_tile_idx, i, j]  # (c_in_tile, c_out_tile)
                            # Load the input tile for the current input channel tile idx, output row, filter position
                            x_tile = X[img, c_in_tile_idx * c_in_tile: (c_in_tile_idx + 1) * c_in_tile, i + out_row, j:j + out_width]  # (c_in_tile, out_width)
                            # Matmul the weight tile and input tile, and accumulate the result in row_out
                            row_out += w_tile.T @ x_tile

                # Load and add the bias to the row_out based on the current output channel tile idx
                b = bias[c_out_tile_idx * c_out_tile: (c_out_tile_idx + 1) * c_out_tile]
                output = row_out + b[:, np.newaxis]

                # Store the output
                X_out[img, c_out_tile_idx * c_out_tile: (c_out_tile_idx + 1) * c_out_tile, out_row, :] = output

    return X_out
