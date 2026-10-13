# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Triton kernel for euclidean distance transform (EDT)"""

import torch
import triton
import triton.language as tl
import numpy as np
from scipy.ndimage import distance_transform_edt

"""
Disclaimer: This implementation is not meant to be extremely efficient. A CUDA kernel would likely be more efficient.
Even in Triton, there may be more suitable algorithms.

The goal of this kernel is to mimic cv2.distanceTransform(input, cv2.DIST_L2, 0).
Recall that the euclidean distance transform (EDT) calculates the L2 distance to the closest zero pixel for each pixel of the source image.

For images of size NxN, the naive algorithm would be to compute pairwise distances between every pair of points, leading to a O(N^4) algorithm, which is obviously impractical.
One can do better using the following approach:
- First, compute the distance to the closest point in the same row. We can write it as Row_EDT[i,j] = min_k (sqrt((k-j)^2) if input[i,k]==0 else +infinity). With a naive implementation, this step has a O(N^3) complexity
- Then, because of triangular inequality, we notice that the EDT for a given location [i,j] is the min of the row EDTs in the same column. EDT[i,j] = min_k Row_EDT[k, j]. This is also O(N^3)

Overall, this algorithm is quite amenable to parallelization, and has a complexity O(N^3). Can we do better?

It turns out that we can leverage the structure of the L2 distance (nice and convex) to find the minimum in a more efficient way.
We follow the algorithm from "Distance Transforms of Sampled Functions" (https://cs.brown.edu/people/pfelzens/papers/dt-final.pdf), which is also what's implemented in opencv

For a single dimension EDT, we can compute the EDT of an arbitrary function F, that we discretize over the grid. Note that for the binary EDT that we're interested in, we can set F(i,j) = 0 if input[i,j]==0 else +infinity
For now, we'll compute the EDT squared, and will take the sqrt only at the very end.
The basic idea is that each point at location i spawns a parabola around itself, with a bias equal to F(i). So specifically, we're looking at the parabola (x - i)^2 + F(i)
When we're looking for the row EDT at location j, we're effectively looking for min_i (x-i)^2 + F(i). In other word we want to find the lowest parabola at location j.

To do this efficiently, we need to maintain the lower envelope of the union of parabolas. This can be constructed on the fly using a sort of stack approach:
 - every time we want to add a new parabola, we check if it may be covering the current right-most parabola. If so, then that parabola was useless, so we can pop it from the stack
 - repeat until we can't find any more parabola to pop. Then push the new one.

This algorithm runs in O(N) for a single row, so overall O(N^2) when applied to all rows
Similarly as before, we notice that we can decompose the algorithm for rows and columns, leading to an overall run-time of O(N^2)

This algorithm is less suited for to GPUs, since the one-dimensional EDT computation is quite sequential in nature. However, we can parallelize over batch and row dimensions.
In Triton, things are particularly bad at the moment, since there is no support for reading/writing to the local memory at a specific index (a local gather is coming soon, see https://github.com/triton-lang/triton/issues/974, but no mention of writing, ie scatter)
One could emulate these operations with masking, but in initial tests, it proved to be worst than naively reading and writing to the global memory. My guess is that the cache is compensating somewhat for the repeated single-point accesses.


The timing obtained on a H100 for a random batch of masks of dimension 256 x 1024 x 1024 are as follows:
- OpenCV: 1780ms (including round-trip to cpu, but discounting the fact that it introduces a synchronization point)
- triton, O(N^3) algo: 627ms
- triton, O(N^2) algo: 322ms

Overall, despite being quite naive, this implementation is roughly 5.5x faster than the openCV cpu implem

--------------------------------------------------------------------------------
RTX 5090 (Blackwell, sm_120) compatibility patch
--------------------------------------------------------------------------------
Triton's CUDA kernel compiler does not yet emit valid device code for sm_120
(Blackwell). Launching the Triton kernel below on such GPUs raises
"CUDA error: device kernel image is invalid" at runtime (not at import time).

To keep this workflow usable, `edt_triton()` now tries the original Triton
kernel first (so it will automatically resume using the fast GPU path once
Triton ships sm_120 support), and transparently falls back to an equivalent
CPU implementation (scipy.ndimage.distance_transform_edt) if the Triton
launch fails. The result is numerically equivalent, just slower
(~15-30ms per call on a 1024x1024 mask vs <1ms on GPU), which is negligible
compared to the overall video segmentation pipeline runtime.
"""


@triton.jit
def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
    # This is a somewhat verbatim implementation of the efficient 1D EDT algorithm described above
    # It can be applied horizontally or vertically depending if we're doing the first or second stage.
    # It's parallelized across batch+row (or batch+col if horizontal=False)
    # TODO: perhaps the implementation can be revisited if/when local gather/scatter become available in triton
    batch_id = tl.program_id(axis=0)
    if horizontal:
        row_id = tl.program_id(axis=1)
        block_start = (batch_id * height * width) + row_id * width
        length = width
        stride = 1
    else:
        col_id = tl.program_id(axis=1)
        block_start = (batch_id * height * width) + col_id
        length = height
        stride = width

    # This will be the index of the right most parabola in the envelope ("the top of the stack")
    k = 0
    for q in range(1, length):
        # Read the function value at the current location. Note that we're doing a singular read, not very efficient
        cur_input = tl.load(inputs_ptr + block_start + (q * stride))
        # location of the parabola on top of the stack
        r = tl.load(v + block_start + (k * stride))
        # associated boundary
        z_k = tl.load(z + block_start + (k * stride))
        # value of the function at the parabola location
        previous_input = tl.load(inputs_ptr + block_start + (r * stride))
        # intersection between the two parabolas
        s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

        # we'll pop as many parabolas as required
        while s <= z_k and k - 1 >= 0:
            k = k - 1
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

        # Store the new one
        k = k + 1
        tl.store(v + block_start + (k * stride), q)
        tl.store(z + block_start + (k * stride), s)
        if k + 1 < length:
            tl.store(z + block_start + ((k + 1) * stride), 1e9)

    # Last step, we read the envelope to find the min in every location
    k = 0
    for q in range(length):
        while (
            k + 1 < length
            and tl.load(
                z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
            )
            < q
        ):
            k += 1
        r = tl.load(v + block_start + (k * stride))
        d = q - r
        old_value = tl.load(inputs_ptr + block_start + (r * stride))
        tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)


def _edt_triton_gpu(data: torch.Tensor) -> torch.Tensor:
    """Original GPU implementation using the Triton kernel above."""
    B, H, W = data.shape
    data = data.contiguous()

    # Allocate the "function" tensor. Implicitly the function is 0 if data[i,j]==0 else +infinity
    output = torch.where(data, 1e18, 0.0)
    assert output.is_contiguous()

    # Scratch tensors for the parabola stacks
    parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
    parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    # Grid size (number of blocks)
    grid = (B, H)

    # Launch initialization kernel
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=True,
    )

    # reset the parabola stacks
    parabola_loc.zero_()
    parabola_inter[:, :, 0] = -1e18
    parabola_inter[:, :, 1] = 1e18

    grid = (B, W)
    edt_kernel[grid](
        output.clone(),
        output,
        parabola_loc,
        parabola_inter,
        H,
        W,
        horizontal=False,
    )
    # don't forget to take sqrt at the end
    return output.sqrt()


def _edt_cpu_fallback(data: torch.Tensor) -> torch.Tensor:
    """
    CPU fallback for GPUs where the Triton EDT kernel is unsupported
    (e.g. RTX 5090 / Blackwell sm_120).

    Numerically equivalent to _edt_triton_gpu / cv2.distanceTransform(input, cv2.DIST_L2, 0):
    for every pixel, the distance to the closest pixel where data == 0.
    """
    original_device = data.device
    mask_np = data.detach().to("cpu").numpy()
    B = mask_np.shape[0]
    out = np.empty(mask_np.shape, dtype=np.float32)
    for b in range(B):
        # distance_transform_edt(x) returns, for every position where x is False,
        # the distance to the nearest True position (and 0 where x is already True).
        # Passing (mask_np[b] == 0) as x makes "True" == "original pixel is 0",
        # which reproduces cv2's "distance to closest zero pixel" semantics.
        bg_mask = mask_np[b] == 0
        out[b] = distance_transform_edt(bg_mask)
    return torch.from_numpy(out).to(device=original_device, dtype=torch.float32)


# Set to True once we detect that the Triton kernel is unusable on this GPU,
# so subsequent calls skip straight to the CPU fallback instead of repeatedly
# paying for a failed CUDA launch.
_TRITON_EDT_BROKEN = False


def edt_triton(data: torch.Tensor):
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.

    Returns:
        A tensor of the same shape as data containing the EDT.
        It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)

    Note:
        Transparently falls back to a CPU (scipy) implementation if the Triton
        GPU kernel fails to launch (e.g. unsupported GPU architecture such as
        RTX 5090 / Blackwell sm_120 with Triton versions that don't yet emit
        valid sm_120 device code).
    """
    global _TRITON_EDT_BROKEN
    assert data.dim() == 3
    assert data.is_cuda

    if not _TRITON_EDT_BROKEN:
        try:
            return _edt_triton_gpu(data)
        except Exception as e:
            _TRITON_EDT_BROKEN = True
            print(
                f"[sam3][edt] Triton EDT kernel failed ({type(e).__name__}: {e}); "
                "falling back to CPU (scipy) implementation for this and all "
                "subsequent calls. This is expected on GPUs not yet supported "
                "by the installed Triton version (e.g. RTX 5090 / Blackwell)."
            )

    return _edt_cpu_fallback(data)
