# Fork Notes — sm_120 (RTX 5090 Blackwell) Compatibility Patches

This repository is a **patched copy** of
[yolain/ComfyUI-Easy-Sam3](https://github.com/yolain/ComfyUI-Easy-Sam3)
(upstream `88fe578`, Apache-2.0, all credit to the original author).

Upstream is pinned at commit `88fe578a1a5e03d95281197303d5d3a73fd5a089` (2025-12-23).
Only three files are modified; everything else is identical to upstream.

## Why

SAM3 ships Triton kernels that fail on Blackwell GPUs (`sm_120`, e.g. RTX 5090)
at runtime with `CUDA error: device kernel image is invalid`. Triton does not
yet emit valid device code for that architecture.

Each patched kernel now **tries the original Triton/GPU path first** and falls
back to a numerically equivalent CPU implementation on failure — so the nodes
keep working on sm_120 today and automatically resume the fast GPU path once
Triton ships Blackwell support. The same fallbacks also help on `sm_89`
(4090-class Ada) where some kernels are unavailable.

## Changed files

| File | Patch |
|---|---|
| `sam3/model/edt.py` | `edt_triton()` → try Triton first, fall back to `scipy.ndimage.distance_transform_edt` (CPU). ~15-30ms/call on 1024² mask, negligible vs whole pipeline. |
| `sam3/perflib/nms.py` | Skip Triton NMS on invalid-kernel failure; tiny IoU matrix computed on CPU (`generic_nms_cpu`). |
| `sam3/perflib/connected_components.py` | CPU single-mask connected-component fallback when the accelerated path is unavailable. |

## New dependency

```
scipy    # only used by the CPU fallback paths
```

If `scipy` is missing the nodes still load, but on sm_120 the patched ops will
raise at run time — install it: `pip install scipy`.

## Verified

- RTX 4090 (48G, sm_89): `easy sam3ImageSegmentation` prompt="person" ✅,
  `easy sam3VideoSegmentation` on a 107-frame clip ✅ (mask exported per-frame)
- RTX 5090 (32G, sm_120): model `sam3.pt` staged; node execution pending
  container re-deployment.

## Model weights

Download `sam3.pt` (~3.45 GB) into `ComfyUI/models/sam3/`.
Weights are **not** part of this repo (Meta SAM3 license).
