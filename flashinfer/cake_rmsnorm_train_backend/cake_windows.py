"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Program selection by token count of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (SM100 / SM103 / SM107), rendered from the registry
of this package (:mod:`.cake_jit`) by its exporter; do not edit.  ``PROGRAM_WINDOWS``
is the table ``cake_jit.program_windows`` derives from the registry's ``MODULES`` and
``ROUTES`` at call time; ``EDGES`` the measured outcome at every window edge; ``NOTES``
the contract of the outputs across the windows; ``SECTIONS`` the text the host module
appends to the forward and backward docstrings.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 2048, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 2048, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 512, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (512, 8192, "cake_rmsnorm_train_9b101716fe0f7404646a"),
            (8192, 24576, "cake_rmsnorm_train_7190dfb41d8917e9476e"),
            (24576, None, "cake_rmsnorm_train_3d5eb81d71ecc25174da"),
        ),
        "fwd_w32_h256": (
            (0, 512, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (512, 8192, "cake_rmsnorm_train_18a2e439aefc9d0db14a"),
            (8192, 24576, "cake_rmsnorm_train_80d5e0c2592cf0890944"),
            (24576, None, "cake_rmsnorm_train_d3224c98e1fdf8bf8bd8"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 24576, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (24576, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 24576, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (24576, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_fd1097c45c1e328871fd"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, None, "cake_rmsnorm_train_1defffdf2712cae36784"),
        ),
        "fwd_h6144": (
            (0, 768, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (768, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 768, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (768, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": ((0, None, "cake_rmsnorm_train_34a86aa5963ced91a2ff"),),
        "bwd_w32_h256": ((0, None, "cake_rmsnorm_train_46f51338c6f926913fd1"),),
        "bwd_h512": ((0, None, "cake_rmsnorm_train_6643d57c6d06d97591ea"),),
        "bwd_w32_h512": ((0, None, "cake_rmsnorm_train_e4fcc22eaf0f41c41733"),),
        "bwd_h1536": ((0, None, "cake_rmsnorm_train_c1fd076f1d546c738faa"),),
        "bwd_w32_h1536": ((0, None, "cake_rmsnorm_train_e54e7b6e14559821fa59"),),
        "bwd_h2048": ((0, None, "cake_rmsnorm_train_22dd3e26be3e6e967b8d"),),
        "bwd_w32_h2048": ((0, None, "cake_rmsnorm_train_c01d393ae1c5a5345fa7"),),
        "bwd_h4096": ((0, None, "cake_rmsnorm_train_840520351132359f4477"),),
        "bwd_w32_h4096": ((0, None, "cake_rmsnorm_train_74673c6e9720e22a03e1"),),
        "bwd_h5120": ((0, None, "cake_rmsnorm_train_04c86dbf2404c816c348"),),
        "bwd_w32_h5120": ((0, None, "cake_rmsnorm_train_3b125d2bc50aabcdc8ed"),),
        "bwd_h6144": ((0, None, "cake_rmsnorm_train_5b1b62335a0652c9f3b1"),),
        "bwd_residual_h6144": ((0, None, "cake_rmsnorm_train_1d84f082dd3ae27ed963"),),
        "bwd_w32_h6144": ((0, None, "cake_rmsnorm_train_72d0946fe83c0c605957"),),
        "bwd_w32_residual_h6144": ((0, None, "cake_rmsnorm_train_b68265b368a076a73620"),),
    },
    "sm_103a": {
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_c5a8099e177e463daab9"),
            (1024, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_01f0ebf3db9fa2c0884e"),
            (1024, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 512, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (512, 8192, "cake_rmsnorm_train_9b101716fe0f7404646a"),
            (8192, None, "cake_rmsnorm_train_7190dfb41d8917e9476e"),
        ),
        "fwd_w32_h256": (
            (0, 512, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (512, 8192, "cake_rmsnorm_train_18a2e439aefc9d0db14a"),
            (8192, None, "cake_rmsnorm_train_80d5e0c2592cf0890944"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 3072, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (3072, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 3072, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (3072, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_fd1097c45c1e328871fd"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 16132, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (16132, None, "cake_rmsnorm_train_1defffdf2712cae36784"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": ((0, None, "cake_rmsnorm_train_34a86aa5963ced91a2ff"),),
        "bwd_w32_h256": ((0, None, "cake_rmsnorm_train_46f51338c6f926913fd1"),),
        "bwd_h512": ((0, None, "cake_rmsnorm_train_6643d57c6d06d97591ea"),),
        "bwd_w32_h512": ((0, None, "cake_rmsnorm_train_e4fcc22eaf0f41c41733"),),
        "bwd_h1536": ((0, None, "cake_rmsnorm_train_c1fd076f1d546c738faa"),),
        "bwd_w32_h1536": ((0, None, "cake_rmsnorm_train_e54e7b6e14559821fa59"),),
        "bwd_h2048": ((0, None, "cake_rmsnorm_train_22dd3e26be3e6e967b8d"),),
        "bwd_w32_h2048": ((0, None, "cake_rmsnorm_train_c01d393ae1c5a5345fa7"),),
        "bwd_h4096": ((0, None, "cake_rmsnorm_train_840520351132359f4477"),),
        "bwd_w32_h4096": ((0, None, "cake_rmsnorm_train_74673c6e9720e22a03e1"),),
        "bwd_h5120": ((0, None, "cake_rmsnorm_train_04c86dbf2404c816c348"),),
        "bwd_w32_h5120": ((0, None, "cake_rmsnorm_train_3b125d2bc50aabcdc8ed"),),
        "bwd_h6144": ((0, None, "cake_rmsnorm_train_5b1b62335a0652c9f3b1"),),
        "bwd_residual_h6144": ((0, None, "cake_rmsnorm_train_1d84f082dd3ae27ed963"),),
        "bwd_w32_h6144": ((0, None, "cake_rmsnorm_train_72d0946fe83c0c605957"),),
        "bwd_w32_residual_h6144": ((0, None, "cake_rmsnorm_train_b68265b368a076a73620"),),
    },
    "sm_107a": {
        "fwd_h1536": (
            (0, 2048, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (2048, 12288, "cake_rmsnorm_train_4cdc4c495b3f59ff5069"),
            (12288, None, "cake_rmsnorm_train_682675bda578f5b7a312"),
        ),
        "fwd_w32_h1536": (
            (0, 2048, "cake_rmsnorm_train_21d3416baa877808371e"),
            (2048, 12288, "cake_rmsnorm_train_92cf7b87f1ebffb1b12b"),
            (12288, None, "cake_rmsnorm_train_a30e5a3b69b7c183992f"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 8192, "cake_rmsnorm_train_a4da85136fba4588239f"),
            (8192, 12288, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (12288, 16384, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (16384, None, "cake_rmsnorm_train_bde56bd5a68c493f07a2"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 8192, "cake_rmsnorm_train_a72c6ece1900f748eaa2"),
            (8192, 12288, "cake_rmsnorm_train_595f506004038a65cc63"),
            (12288, 16384, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
            (16384, None, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
        ),
        "fwd_h256": (
            (0, 512, "cake_rmsnorm_train_c5ca409f44e169148301"),
            (512, 2048, "cake_rmsnorm_train_9b101716fe0f7404646a"),
            (2048, None, "cake_rmsnorm_train_4943718b999b78b73537"),
        ),
        "fwd_w32_h256": (
            (0, 512, "cake_rmsnorm_train_8f2756937eaac6b02502"),
            (512, 2048, "cake_rmsnorm_train_18a2e439aefc9d0db14a"),
            (2048, None, "cake_rmsnorm_train_1722060608242420167e"),
        ),
        "fwd_h4096": (
            (0, 512, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (512, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 4096, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (4096, 20480, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (20480, 49152, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (49152, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 512, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (512, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 4096, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (4096, 20480, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (20480, 49152, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (49152, None, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
        ),
        "fwd_h512": (
            (0, 256, "cake_rmsnorm_train_7673978dc698790054d2"),
            (256, 6144, "cake_rmsnorm_train_2a42e5c02b240f8709b1"),
            (6144, None, "cake_rmsnorm_train_60f47c5d00570ecf7740"),
        ),
        "fwd_w32_h512": (
            (0, 256, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (256, 6144, "cake_rmsnorm_train_79fecc7016e44e940733"),
            (6144, None, "cake_rmsnorm_train_61accbcc02891615f286"),
        ),
        "fwd_h5120": (
            (0, 512, "cake_rmsnorm_train_fd1097c45c1e328871fd"),
            (512, 1152, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (1152, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, 20480, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (20480, 49152, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (49152, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 512, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (512, 1152, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (1152, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 20480, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (20480, 49152, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (49152, None, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, 4096, "cake_rmsnorm_train_cf1d05546951bdbd9f5d"),
            (4096, None, "cake_rmsnorm_train_e7f4221a7c26b9dfdd23"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 4096, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (4096, None, "cake_rmsnorm_train_fdfa500c83d95df0f5c1"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_e646bbdbd852884953da"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_6325c474ad9955a319e9"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": ((0, None, "cake_rmsnorm_train_34a86aa5963ced91a2ff"),),
        "bwd_w32_h256": ((0, None, "cake_rmsnorm_train_46f51338c6f926913fd1"),),
        "bwd_h512": ((0, None, "cake_rmsnorm_train_6643d57c6d06d97591ea"),),
        "bwd_w32_h512": ((0, None, "cake_rmsnorm_train_e4fcc22eaf0f41c41733"),),
        "bwd_h1536": ((0, None, "cake_rmsnorm_train_c1fd076f1d546c738faa"),),
        "bwd_w32_h1536": ((0, None, "cake_rmsnorm_train_e54e7b6e14559821fa59"),),
        "bwd_h2048": ((0, None, "cake_rmsnorm_train_22dd3e26be3e6e967b8d"),),
        "bwd_w32_h2048": ((0, None, "cake_rmsnorm_train_c01d393ae1c5a5345fa7"),),
        "bwd_h4096": ((0, None, "cake_rmsnorm_train_840520351132359f4477"),),
        "bwd_w32_h4096": ((0, None, "cake_rmsnorm_train_74673c6e9720e22a03e1"),),
        "bwd_h5120": ((0, None, "cake_rmsnorm_train_04c86dbf2404c816c348"),),
        "bwd_w32_h5120": ((0, None, "cake_rmsnorm_train_3b125d2bc50aabcdc8ed"),),
        "bwd_h6144": ((0, None, "cake_rmsnorm_train_5b1b62335a0652c9f3b1"),),
        "bwd_residual_h6144": ((0, None, "cake_rmsnorm_train_1d84f082dd3ae27ed963"),),
        "bwd_w32_h6144": ((0, None, "cake_rmsnorm_train_72d0946fe83c0c605957"),),
        "bwd_w32_residual_h6144": ((0, None, "cake_rmsnorm_train_b68265b368a076a73620"),),
    },
}

# Per architecture and route: the upper edge of every window but the last and the measured
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (the
# forward's one-ulp differences), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_h256": ((512, "unmeasured"), (8192, "unmeasured"), (24576, "unmeasured")),
        "fwd_w32_h256": (
            (512, "unmeasured"),
            (8192, "unmeasured"),
            (24576, "unmeasured"),
        ),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_h5120": ((640, "unmeasured"), (6144, "unmeasured")),
        "fwd_w32_h5120": ((640, "unmeasured"), (6144, "unmeasured")),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": (
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (6144, "unmeasured"),
        ),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h256": (),
        "bwd_w32_h256": (),
        "bwd_h512": (),
        "bwd_w32_h512": (),
        "bwd_h1536": (),
        "bwd_w32_h1536": (),
        "bwd_h2048": (),
        "bwd_w32_h2048": (),
        "bwd_h4096": (),
        "bwd_w32_h4096": (),
        "bwd_h5120": (),
        "bwd_w32_h5120": (),
        "bwd_h6144": (),
        "bwd_residual_h6144": (),
        "bwd_w32_h6144": (),
        "bwd_w32_residual_h6144": (),
    },
    "sm_103a": {
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_h256": ((512, "unmeasured"), (8192, "unmeasured")),
        "fwd_w32_h256": ((512, "unmeasured"), (8192, "unmeasured")),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_h5120": ((640, "unmeasured"), (6144, "unmeasured")),
        "fwd_w32_h5120": (
            (640, "unmeasured"),
            (6144, "unmeasured"),
            (16132, "unmeasured"),
        ),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "unmeasured"), (6144, "unmeasured")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h256": (),
        "bwd_w32_h256": (),
        "bwd_h512": (),
        "bwd_w32_h512": (),
        "bwd_h1536": (),
        "bwd_w32_h1536": (),
        "bwd_h2048": (),
        "bwd_w32_h2048": (),
        "bwd_h4096": (),
        "bwd_w32_h4096": (),
        "bwd_h5120": (),
        "bwd_w32_h5120": (),
        "bwd_h6144": (),
        "bwd_residual_h6144": (),
        "bwd_w32_h6144": (),
        "bwd_w32_residual_h6144": (),
    },
    "sm_107a": {
        "fwd_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_w32_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_h2048": ((1024, "ulp"), (8192, "same"), (12288, "same"), (16384, "same")),
        "fwd_w32_h2048": (
            (1024, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_h256": ((512, "unmeasured"), (2048, "unmeasured")),
        "fwd_w32_h256": ((512, "unmeasured"), (2048, "unmeasured")),
        "fwd_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h4096": (
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_h512": ((256, "same"), (6144, "same")),
        "fwd_w32_h512": ((256, "same"), (6144, "same")),
        "fwd_h5120": (
            (512, "unmeasured"),
            (1152, "unmeasured"),
            (6144, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_w32_h5120": (
            (512, "unmeasured"),
            (1152, "unmeasured"),
            (6144, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_h6144": ((1024, "same"), (4096, "same")),
        "fwd_w32_h6144": ((1024, "same"), (4096, "same")),
        "fwd_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "bwd_h256": (),
        "bwd_w32_h256": (),
        "bwd_h512": (),
        "bwd_w32_h512": (),
        "bwd_h1536": (),
        "bwd_w32_h1536": (),
        "bwd_h2048": (),
        "bwd_w32_h2048": (),
        "bwd_h4096": (),
        "bwd_w32_h4096": (),
        "bwd_h5120": (),
        "bwd_w32_h5120": (),
        "bwd_h6144": (),
        "bwd_residual_h6144": (),
        "bwd_w32_h6144": (),
        "bwd_w32_residual_h6144": (),
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

NOTES = """
Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and across a forward edge between programs of different per-row thread groups ``rstd``
may move by one FP32 ulp and ``y`` by one BF16 ulp on a few rows.  The backward
(``deterministic=False``) computes ``dx`` row by row as the deterministic backward does
and adds the FP32 ``dw`` partials of its CTAs with atomics in a scheduling-dependent
order: ``dw`` is not bitwise reproducible, even for a fixed ``T``.
"""

FORWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and across a forward edge between programs of different per-row thread groups ``rstd``
may move by one FP32 ulp and ``y`` by one BF16 ulp on a few rows.  The backward
(``deterministic=False``) computes ``dx`` row by row as the deterministic backward does
and adds the FP32 ``dw`` partials of its CTAs with atomics in a scheduling-dependent
order: ``dw`` is not bitwise reproducible, even for a fixed ``T``.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it, ``~``
the forward's one-ulp differences, ``?`` not yet measured.

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

* ``sm_100a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_100a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 2048] 780b9833 ~
  (2048, 4096] b44107e0 ~ (4096, inf) 780b9833
* ``sm_100a`` ``fwd_h256``: (0, 512] 0489c784 ? (512, 8192] 9b101716 ?
  (8192, 24576] 7190dfb4 ? (24576, inf) 3d5eb81d
* ``sm_100a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_100a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 24576] 3d20ab4a = (24576, inf) 3430e5d9
* ``sm_100a`` ``fwd_h5120``: (0, 640] fd1097c4 ? (640, 6144] b5e173f5 ?
  (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e ? (768, 1024] 2763cd5e ?
  (1024, 6144] 3d619cd1 ? (6144, inf) 5f89e5fd
* ``sm_100a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
* ``sm_103a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_103a`` ``fwd_h2048``: (0, 1024] c5a8099e = (1024, 4096] b44107e0 ~
  (4096, inf) 780b9833
* ``sm_103a`` ``fwd_h256``: (0, 512] 0489c784 ? (512, 8192] 9b101716 ?
  (8192, inf) 7190dfb4
* ``sm_103a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 3072] d447b7f6 = (3072, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_103a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 16384] 3d20ab4a = (16384, inf) 3430e5d9
* ``sm_103a`` ``fwd_h5120``: (0, 640] fd1097c4 ? (640, 6144] b5e173f5 ?
  (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 640] 1defffdf ? (640, 6144] 9ecb62c2 ?
  (6144, 16132] 9ecb62c2 ? (16132, inf) 1defffdf
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e ? (1024, 6144] 2763cd5e ?
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
* ``sm_107a`` ``fwd_h1536``: (0, 2048] 1ea88523 ~ (2048, 12288] 4cdc4c49 =
  (12288, inf) 682675bd
* ``sm_107a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 8192] a4da8513 =
  (8192, 12288] 0ad4d53c = (12288, 16384] 0ad4d53c = (16384, inf) bde56bd5
* ``sm_107a`` ``fwd_w32_h2048``: (0, 1024] 0da8e18d ? (1024, 8192] a72c6ece ?
  (8192, 12288] 595f5060 ? (12288, 16384] 40210950 ? (16384, inf) 40210950
* ``sm_107a`` ``fwd_h256``: (0, 512] c5ca409f ? (512, 2048] 9b101716 ?
  (2048, inf) 4943718b
* ``sm_107a`` ``fwd_h4096``: (0, 512] 8392d1a0 ~ (512, 1024] fd913ecd =
  (1024, 4096] d447b7f6 = (4096, 20480] bf933e16 = (20480, 49152] bf933e16 =
  (49152, inf) fd913ecd
* ``sm_107a`` ``fwd_w32_h4096``: (0, 512] 4371270e ? (512, 1024] f0db3068 ?
  (1024, 4096] f9f3423b ? (4096, 20480] f9f3423b ? (20480, 49152] f9f3423b ?
  (49152, inf) f9f3423b
* ``sm_107a`` ``fwd_h512``: (0, 256] 7673978d = (256, 6144] 2a42e5c0 =
  (6144, inf) 60f47c5d
* ``sm_107a`` ``fwd_h5120``: (0, 512] fd1097c4 ? (512, 1152] 5eef28cd ?
  (1152, 6144] b5e173f5 ? (6144, 20480] 044d4a2a ? (20480, 49152] 044d4a2a ?
  (49152, inf) 5eef28cd
* ``sm_107a`` ``fwd_w32_h5120``: (0, 512] ba28d2c0 ? (512, 1152] 1defffdf ?
  (1152, 6144] 9ecb62c2 ? (6144, 20480] b49de9b4 ? (20480, 49152] b49de9b4 ?
  (49152, inf) b49de9b4
* ``sm_107a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, 4096] cf1d0554 =
  (4096, inf) e7f4221a
* ``sm_107a`` ``fwd_residual_h6144``: (0, 256] e646bbdb ~ (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
"""

BACKWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and across a forward edge between programs of different per-row thread groups ``rstd``
may move by one FP32 ulp and ``y`` by one BF16 ulp on a few rows.  The backward
(``deterministic=False``) computes ``dx`` row by row as the deterministic backward does
and adds the FP32 ``dw`` partials of its CTAs with atomics in a scheduling-dependent
order: ``dw`` is not bitwise reproducible, even for a fixed ``T``.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it, ``~``
the forward's one-ulp differences, ``?`` not yet measured.

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

* ``sm_100a`` ``bwd_h256``: (0, inf) 34a86aa5
* ``sm_100a`` ``bwd_h512``: (0, inf) 6643d57c
* ``sm_100a`` ``bwd_h1536``: (0, inf) c1fd076f
* ``sm_100a`` ``bwd_h2048``: (0, inf) 22dd3e26
* ``sm_100a`` ``bwd_h4096``: (0, inf) 84052035
* ``sm_100a`` ``bwd_h5120``: (0, inf) 04c86dbf
* ``sm_100a`` ``bwd_h6144``: (0, inf) 5b1b6233
* ``sm_100a`` ``bwd_residual_h6144``: (0, inf) 1d84f082
* ``sm_103a`` ``bwd_h256``: (0, inf) 34a86aa5
* ``sm_103a`` ``bwd_h512``: (0, inf) 6643d57c
* ``sm_103a`` ``bwd_h1536``: (0, inf) c1fd076f
* ``sm_103a`` ``bwd_h2048``: (0, inf) 22dd3e26
* ``sm_103a`` ``bwd_h4096``: (0, inf) 84052035
* ``sm_103a`` ``bwd_h5120``: (0, inf) 04c86dbf
* ``sm_103a`` ``bwd_h6144``: (0, inf) 5b1b6233
* ``sm_103a`` ``bwd_residual_h6144``: (0, inf) 1d84f082
* ``sm_107a`` ``bwd_h256``: (0, inf) 34a86aa5
* ``sm_107a`` ``bwd_h512``: (0, inf) 6643d57c
* ``sm_107a`` ``bwd_h1536``: (0, inf) c1fd076f
* ``sm_107a`` ``bwd_h2048``: (0, inf) 22dd3e26
* ``sm_107a`` ``bwd_h4096``: (0, inf) 84052035
* ``sm_107a`` ``bwd_h5120``: (0, inf) 04c86dbf
* ``sm_107a`` ``bwd_h6144``: (0, inf) 5b1b6233
* ``sm_107a`` ``bwd_residual_h6144``: (0, inf) 1d84f082
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "BACKWARD_SECTION",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
]
