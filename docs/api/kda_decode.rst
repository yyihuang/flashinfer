.. _apikda_decode:

flashinfer.kda_decode
=====================

Key-Driven Attention (KDA) decode API. ``recurrent_kda`` selects an optimized
frozen CUDA backend on B200 for one measured speculative-decode specialization.
Single-token decode and all unsupported speculative shapes, gate modes,
layouts, and features continue to use the CuTe-DSL backend under
``flashinfer.kda_kernels``.

The optimized backend preserves the public BF16 state/checkpoint ABI. Its
measured route is D128/T4/N64/H16/HV32 with a precomputed gate and in-kernel
Q/K L2 normalization. CUDA graphs must warm the JIT module before capture and
provide stable output, state, and speculative metadata tensors; no additional
workspace is required.

.. currentmodule:: flashinfer.kda_decode

.. autosummary::
    :toctree: ../generated

    recurrent_kda
