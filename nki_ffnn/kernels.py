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

    for col_idx in nl.affine_range(HIDDEN_SIZE // nl.tile_size.pmax):
      col_offset = col_idx * nl.tile_size.pmax
      bias_tile = nl.load(b[0:1, col_offset:col_offset + nl.tile_size.pmax])
      for row_idx in nl.affine_range(BATCH_SIZE // nl.tile_size.pmax):
        row_offset = row_idx * nl.tile_size.pmax
        a_tile = nl.load(A[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax])
        out_tile = nl.add(a_tile, bias_tile)
        if act == 'relu':
          out_tile = nl.relu(out_tile)
        nl.store(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax], out_tile)

    if act == 'relu':
      return result

    row_max_buf = nl.ndarray((BATCH_SIZE, 1), dtype=A.dtype, buffer=nl.hbm)
    row_sum_buf = nl.ndarray((BATCH_SIZE, 1), dtype=A.dtype, buffer=nl.hbm)
    num_col_tiles = HIDDEN_SIZE // nl.tile_size.pmax

    for row_idx in nl.affine_range(BATCH_SIZE // nl.tile_size.pmax):
      row_offset = row_idx * nl.tile_size.pmax
      row_max = nl.ndarray((nl.tile_size.pmax, 1), dtype=A.dtype, buffer=nl.sbuf)
      first_tile = nl.load(result[row_offset:row_offset + nl.tile_size.pmax, 0:nl.tile_size.pmax])
      row_max[...] = nl.max(first_tile, axis=[1], keepdims=True)
      for col_idx in nl.sequential_range(1, num_col_tiles):
        col_offset = col_idx * nl.tile_size.pmax
        tile = nl.load(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax])
        tile_max = nl.max(tile, axis=[1], keepdims=True)
        row_max[...] = nl.maximum(row_max[...], tile_max)
      nl.store(row_max_buf[row_offset:row_offset + nl.tile_size.pmax, 0:1], row_max[...])

    for row_idx in nl.affine_range(BATCH_SIZE // nl.tile_size.pmax):
      row_offset = row_idx * nl.tile_size.pmax
      row_max = nl.load(row_max_buf[row_offset:row_offset + nl.tile_size.pmax, 0:1])
      row_sum = nl.ndarray((nl.tile_size.pmax, 1), dtype=A.dtype, buffer=nl.sbuf)
      first_tile = nl.load(result[row_offset:row_offset + nl.tile_size.pmax, 0:nl.tile_size.pmax])
      first_exp_tile = nl.exp(nl.subtract(first_tile, row_max))
      row_sum[...] = nl.sum(first_exp_tile, axis=[1], keepdims=True)
      nl.store(result[row_offset:row_offset + nl.tile_size.pmax, 0:nl.tile_size.pmax], first_exp_tile)
      for col_idx in nl.sequential_range(1, num_col_tiles):
        col_offset = col_idx * nl.tile_size.pmax
        tile = nl.load(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax])
        exp_tile = nl.exp(nl.subtract(tile, row_max))
        tile_sum = nl.sum(exp_tile, axis=[1], keepdims=True)
        row_sum[...] = nl.add(row_sum[...], tile_sum)
        nl.store(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax], exp_tile)
      nl.store(row_sum_buf[row_offset:row_offset + nl.tile_size.pmax, 0:1], row_sum[...])

    for row_idx in nl.affine_range(BATCH_SIZE // nl.tile_size.pmax):
      row_offset = row_idx * nl.tile_size.pmax
      row_sum = nl.load(row_sum_buf[row_offset:row_offset + nl.tile_size.pmax, 0:1])
      for col_idx in nl.affine_range(HIDDEN_SIZE // nl.tile_size.pmax):
        col_offset = col_idx * nl.tile_size.pmax
        tile = nl.load(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax])
        probs_tile = nl.divide(tile, row_sum)
        nl.store(result[row_offset:row_offset + nl.tile_size.pmax, col_offset:col_offset + nl.tile_size.pmax], probs_tile)

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
  XT = nki_transpose(X)
  z1 = nki_matmul(XT, W1)
  a1 = nki_bias_add_act(z1, b1, act='relu')

  # Layer 2 (output)
  a1T = nki_transpose(a1)
  z2 = nki_matmul(a1T, W2)
  probs = nki_bias_add_act(z2, b2, act='softmax')

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
  probs = nki_forward(X, W1, b1, W2, b2, matmul_kernel=matmul_kernel)
  BATCH_SIZE, OUTPUT_SIZE = probs.shape
  predictions = nl.ndarray((BATCH_SIZE,), dtype=np.int32, buffer=nl.hbm)

  for row_idx in nl.affine_range(BATCH_SIZE // nl.tile_size.pmax):
    row_offset = row_idx * nl.tile_size.pmax
    probs_tile = nl.load(probs[row_offset:row_offset + nl.tile_size.pmax, 0:OUTPUT_SIZE])
    max_vals = nisa.max8(src=probs_tile)
    max_indices = nisa.nc_find_index8(data=probs_tile, vals=max_vals)
    argmax = nl.copy(max_indices[0:nl.tile_size.pmax, 0], dtype=np.int32)
    nl.store(predictions[row_offset:row_offset + nl.tile_size.pmax], argmax)

  return predictions
