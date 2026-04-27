import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from utils import BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from matmul_kernels import nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

@nki.jit
def nki_transpose(in_tensor):
    """NKI kernel to transpose a 2D tensor.

    Args:
        in_tensor: an input tensor of shape [#rows, #cols]

    Returns:
        out_tensor: an output (transposed) tensor of shape [#cols, #rows]
    """
    i_rows, i_cols = in_tensor.shape
    o_rows, o_cols = i_cols, i_rows
    out_tensor = nl.ndarray((o_rows, o_cols), dtype=in_tensor.dtype, buffer=nl.hbm)
    for row_idx in nl.affine_range(in_tensor.shape[0] // 128):
      for col_idx in nl.affine_range(in_tensor.shape[1] // nl.tile_size.pmax):
        row_offset = row_idx * 128
        col_offset = col_idx * nl.tile_size.pmax
        transpose = nl.load_transpose2d(in_tensor[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax])
        nl.store(out_tensor[col_offset:col_offset + nl.tile_size.pmax, row_offset:row_offset + 128], transpose)
    
    return out_tensor

@nki.jit
def nki_bias_add_act(A, b, act='relu'):
    """NKI kernel to add a bias vector to each row of a 2D tensor, and apply activation.

    Args:
        A: an input tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
        b: a bias vector of shape [1, HIDDEN_SIZE]
        act: an activation function to apply (e.g., 'relu', 'softmax')
    Returns:
        result: the resulting output tensor of shape [BATCH_SIZE, HIDDEN_SIZE]
    """
    # Gather input shapes
    BATCH_SIZE, HIDDEN_SIZE = A.shape
    _, HIDDEN_SIZE_ = b.shape
    assert HIDDEN_SIZE == HIDDEN_SIZE_, "A and b must have the same HIDDEN_SIZE"

    # Create an output tensor
    result = nl.ndarray((BATCH_SIZE, HIDDEN_SIZE), dtype=A.dtype, buffer=nl.hbm)
    # YOUR CODE HERE
    for col_idx in nl.affine_range(A.shape[1] // nl.tile_size.pmax):
      col_offset = col_idx * nl.tile_size.pmax
      bias = nl.load(b[0:1, col_offset:col_offset + nl.tile_size.pmax])
      for row_idx in nl.affine_range(A.shape[0] // 127):
        row_offset = row_idx * 127
        tile = nl.load(A[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax])
        tile = nl.add(tile, bias)
        if act == 'relu':
          tile = nl.relu(tile)
        nl.store(result[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax], tile)
    if A.shape[0] % 127 != 0:
      for col_idx in nl.affine_range(A.shape[1] // nl.tile_size.pmax):
          col_offset = col_idx * nl.tile_size.pmax
          tile = nl.load(A[A.shape[0]-A.shape[0] % 127:A.shape[0], col_offset:col_offset + nl.tile_size.pmax])
          bias = nl.load(b[0:1, col_offset:col_offset + nl.tile_size.pmax])
          tile = nl.add(tile, bias)
          if act == 'relu':
            tile = nl.relu(tile)
          nl.store(result[A.shape[0]-A.shape[0] % 127:A.shape[0], col_offset:col_offset + nl.tile_size.pmax], tile)
    if act == 'relu':
      return result
    # softmax part
    running_buffer = nl.ndarray((1, A.shape[0]), dtype=A.dtype, buffer=nl.hbm)
    nl.store(running_buffer, value = 0.0)
    for row_idx in nl.affine_range(A.shape[0] // 128):
      row_offset = row_idx * 128
      rb = nl.load(running_buffer[0, row_offset:row_offset + 128])
      for col_idx in nl.affine_range(A.shape[1] // (nl.tile_size.pmax - 2)):
        col_offset = col_idx * (nl.tile_size.pmax - 2)
        tile = nl.load(A[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax - 2])
        cur_max = nl.max(tile, [1])
        rb = nl.maximum(cur_max, rb)
        if col_idx == A.shape[1] // (nl.tile_size.pmax - 2) - 1:
          nl.store(running_buffer[0, row_offset:row_offset + 128], rb)
    for row_idx in nl.affine_range(A.shape[0] // 128):
      row_offset = row_idx * 128
      rb = nl.load(running_buffer[0, row_offset:row_offset + 128])
      for col_idx in nl.affine_range(A.shape[1] // (nl.tile_size.pmax - 3)):
        col_offset = col_idx * (nl.tile_size.pmax - 3)
        row_offset = row_idx * 128
        tile = nl.load(A[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax - 3])
        tile = nl.subtract(tile, rb)
        tile = nl.exp(tile)
        if col_idx == 0:
          running_sum = nl.sum(tile, [1])
        else:
          cur_sum = nl.sum(tile, [1])
          running_sum = nl.add(cur_sum, running_sum)
        nl.store(result[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax - 3], tile)
        if col_idx == A.shape[1] // (nl.tile_size.pmax - 3) - 1:
          nl.store(running_buffer[0, row_offset:row_offset + 128], running_sum)
    for row_idx in nl.affine_range(A.shape[0] // 128):
      row_offset = row_idx * 128
      rb = nl.load(running_buffer[0, row_offset:row_offset + 128])
      for col_idx in nl.affine_range(A.shape[1] // (nl.tile_size.pmax - 2)):
        col_offset = col_idx * (nl.tile_size.pmax - 2)
        row_offset = row_idx * 128
        tile = nl.load(A[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax - 2])
        tile = nl.divide(tile, rb)
        nl.store(result[row_offset:row_offset + 128, col_offset:col_offset + nl.tile_size.pmax - 2], tile) 
    return result   
    # if A.shape[0] % 127 != 0:
    #   for col_idx in nl.affine_range(A.shape[1] // nl.tile_size.pmax):
    #       col_offset = col_idx * nl.tile_size.pmax
    #       tile = nl.load(A[A.shape[0]-A.shape[0] % 127:A.shape[0], col_offset:col_offset + nl.tile_size.pmax])
    #       bias = nl.load(b[0:1, col_offset:col_offset + nl.tile_size.pmax])
    #       tile = nl.add(tile, bias)
    #       nl.store(result[A.shape[0]-A.shape[0] % 127:A.shape[0], col_offset:col_offset + nl.tile_size.pmax], tile)
    return result

@nki.jit
def nki_forward(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel to compute the forward pass of the feedforward neural network with 1 hidden layer.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      probs: the resulting probability output tensor of shape [BATCH_SIZE, OUTPUT_SIZE]
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'
  """
  if matmul_kernel == 'tiled':
    nki_matmul = nki_matmul_tiled_
  elif matmul_kernel == 'hoist_load':
    nki_matmul = nki_matmul_hoist_load_
  elif matmul_kernel == 'block_free_dimension':
    nki_matmul = nki_matmul_block_free_dimension_
  elif matmul_kernel == 'fully_optimized':
    nki_matmul = nki_matmul_fully_optimized_
  else:
    raise ValueError(f"Unsupported matmul kernel: {matmul_kernel}")

  # Layer 1
  # YOUR CODE HERE  

  # Layer 2 (output)
  # YOUR CODE HERE

  return probs


@nki.jit
def nki_predict(
    X,
    W1,
    b1,
    W2,
    b2,
    matmul_kernel='tiled'
):
  """NKI kernel run forward pass and predict the classes of the input tensor.

  Args:
      X: an input tensor of shape [BATCH_SIZE, INPUT_SIZE]
      W1: the weight matrix of shape [INPUT_SIZE, HIDDEN_SIZE]
      b1: the bias vector of shape [HIDDEN_SIZE]
      W2: the weight matrix of shape [HIDDEN_SIZE, OUTPUT_SIZE]
      b2: the bias vector of shape [OUTPUT_SIZE]
  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  
  Option:
      matmul_kernel: the matrix multiplication kernel to use 
        - Options: 'tiled', 'hoist_load', 'block_free_dimension', 'fully_optimized'

  Returns:
      predictions: a 1D tensor of shape [BATCH_SIZE] with the predicted class for each input
  """
  probs = 3# YOUR CODE HERE
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  # YOUR CODE HERE

  return predictions