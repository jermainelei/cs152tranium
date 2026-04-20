import re
import numpy as np

def params_name(params):
    input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params
    dtype_str = np.zeros(1).astype(dtype).dtype.name
    name = f"in{input_channels}_out{output_channels}_filter{filter_size}x{filter_size}_batch{batch_size}_{image_dims[0]}x{image_dims[0]}_{dtype_str}"
    return name

def test_case_params(test_case):
    m = re.match(r"in(\d+)_out(\d+)_filter(\d+)x(\d+)_batch(\d+)_(\d+)x(\d+)_(\w+)", test_case)
    if not m:
        raise ValueError(f"Invalid test case name: {test_case}")
    
    input_channels = int(m.group(1))
    output_channels = int(m.group(2))
    filter_size = int(m.group(3))
    batch_size = int(m.group(5))
    image_dims = (int(m.group(6)), int(m.group(7)))
    dtype_str = m.group(8)
    if dtype_str == "float16":
        dtype = np.float16
    elif dtype_str == "float32":
        dtype = np.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype_str} for test case: {test_case}")

    params = (input_channels, output_channels, filter_size, batch_size, image_dims, dtype)

    return params

dtype_tol = {
    np.float32: {"rtol": 1e-5, "atol": 1e-8},
    np.float16: {"rtol": 1e-2, "atol": 1e-4},
}

basic_test_cases = {
    "in128_out256_filter3x3_batch4_34x34_float32": 590,
    "in128_out256_filter3x3_batch4_66x66_float16": 700,
    "in256_out256_filter5x5_batch4_36x36_float16": 2500,
}

fleet_test_cases = {
    "in128_out128_filter3x3_batch16_66x66_float16": 1500,
    "in128_out128_filter5x5_batch16_36x36_float16": 1700,
    "in128_out256_filter3x3_batch4_34x34_float32": 590,
    "in128_out256_filter3x3_batch4_66x66_float16": 700,
    "in128_out256_filter3x3_batch16_34x34_float32": 2000,
    "in128_out256_filter5x5_batch4_36x36_float32": 1700,
    "in128_out256_filter5x5_batch16_36x36_float16": 3100,
    "in256_out256_filter3x3_batch4_34x34_float32": 1200,
    "in256_out256_filter3x3_batch16_34x34_float16": 2700,
    "in256_out256_filter5x5_batch4_36x36_float16": 2500,
    "in256_out256_filter5x5_batch16_36x36_float16": 8000,
}
