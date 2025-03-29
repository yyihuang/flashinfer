from typing import Optional

import torch
import triton

from .kernels.sm_constraint_gemm import (
    matmul_kernel_persistent,
)
from .utils import check_device, check_dim, check_input, check_shape


def matmul_persistent(a, b, c=None, alpha=1.0, beta=0.0, num_sms=None):
    """
    GEMM operation with SM constraint by Triton.
    C = alpha * (a @ b.T) + beta * C

    Args:
        a: The first input matrix. Shape: (M, K)
        b: The second input matrix. Shape: (K, N)
        c: The output matrix. Shape: (M, N). In-place operation is supported.
        alpha: The scaling factor for the product of a and b.
        beta: The scaling factor for the output matrix c.
        num_sms: The number of SMs to use for the computation.
    """

    # Check constraints.
    check_input(a)
    # check_input(b) # b can be non-contiguous
    check_input(c)
    check_device([a, b, c])
    check_dim(2, a)
    check_dim(2, b)
    check_dim(2, c)

    assert a.shape[1] == b.shape[0], "Incompatible dimensions between a and b"
    assert a.dtype == b.dtype, "Incompatible dtypes between a and b"
    assert a.shape[0] == c.shape[0], "Incompatible dimensions between a and c"
    assert b.shape[1] == c.shape[1], "Incompatible dimensions between b and c"

    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype) if c is None else c

    # Set num_sms to be 100% of the available SMs
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count 
    num_sms = NUM_SMS if num_sms is None else min(NUM_SMS, num_sms)
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            num_sms,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
        NUM_SMS=num_sms,
    )
    return c
