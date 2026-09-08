# GEMM on GPU: Pipeline Staging and Warp Specialization

This project implements and compares three CUDA versions of a tiled GEMM
(`C = beta*C + alpha*A*B`) on the PolyBench/GPU benchmark harness. All three
solve the identical problem on identical inputs; what changes is the kind of
overlap each one uses to keep the streaming multiprocessor busy.

Two of the three overlap **data movement with arithmetic**: while one tile is
being multiplied, the next tile is already on its way from global memory into
shared memory. That is *load-compute* overlap.

The third one, warp specialization, is **not** load-compute overlap. It
prefetches nothing and moves no data ahead of time. What it overlaps is *two
kinds of arithmetic with each other*: an A100 SM has separate FP32 and FP64
pipelines, and this variant deliberately sends half of its warps down the FP64
path so both sets of units issue at the same time, instead of every warp
queueing for the same FP32 units. That is *compute-compute* overlap, a
different axis entirely, and the distinction is worth holding on to when
reading the three side by side.

## Problem Statement

A tiled GEMM alternates between two phases: load a tile of `A` and `B` into
shared memory, then multiply that tile. Done naively, those phases serialize —
every thread in the block sits at a `__syncthreads()` waiting for the loads to
land, then the load hardware sits idle while the block does its arithmetic. On
a memory-bound kernel that idle time dominates.

Variants 1 and 3 attack that stall directly, from opposite ends — one by hand,
one in hardware. Variant 2 leaves it alone and goes after a different kind of
idleness altogether:

- **Variant 1 - Double Buffering (load-compute overlap):** A software producer/consumer split inside
  the block. Two pairs of shared-memory tile buffers are allocated. Half the
  block's threads (`threadIdx.y < 8`) load tile `t+1` while the other half
  (`threadIdx.y >= 8`) multiplies tile `t` out of the other buffer. The two
  buffer pointers swap every iteration, so the roles alternate between buffers
  rather than being copied around. Because only half the threads compute, each
  compute thread is responsible for two output rows (`val1`, `val2`). This
  needs only one `__syncthreads()` per tile instead of the usual two, and the
  fetch of the next tile is issued before the current tile's arithmetic starts.

- **Variant 2 - Warp Specialization (compute-compute overlap):** This one does
  not address the load/compute stall at all. It spreads warps across the SM's
  *separate* FP32 and FP64 pipelines, which on an A100 are distinct hardware
  units, so that two different classes of arithmetic are in flight together. Two counters live in
  shared memory (`fp32CountInSM`, `fp64CountInSM`). Lane 0 of each warp claims
  a slot with an `atomicCAS` retry loop — FP32 first, falling back to FP64 —
  and broadcasts the outcome to the rest of its warp with
  `__shfl_sync`. Warps holding an FP32 slot accumulate in `float`; warps
  holding an FP64 slot cast to `double` so their multiply-adds are issued to
  the FP64 units, then cast back to `float` on the store. Lane 0 releases the
  slot with `atomicSub` when the dot product finishes. The block is 32x8, so a
  warp is exactly one `threadIdx.y` row and `threadIdx.x` is a real lane id;
  with 8 warps against 4 FP32 + 4 FP64 slots the split comes out 50/50.

- **Variant 3 - Cooperative Pipeline (load-compute overlap):** Does the same overlap as variant 1, but
  in hardware instead of by hand. `cuda::pipeline` with two stages plus
  `cooperative_groups::memcpy_async` issues Ampere `cp.async` copies that move
  data global -> shared directly, without staging through registers and without
  the loading threads blocking on the transfer. The producer loop runs
  `producer_acquire`/`producer_commit` up to `NUM_STAGES` batches ahead of the
  consumer, and `consumer_wait` blocks only when the stage being multiplied is
  genuinely not ready yet. Each thread moves four floats per copy, and the
  stage index (`batch % NUM_STAGES`) provides the same double-buffering effect
  as variant 1's pointer swap.

All three must produce the same result (`Non-Matching ... : 0` against the CPU
reference), and the point is to compare their GPU times on the same matrices.

The matrices are the PolyBench/GPU standard GEMM dataset: 2048 x 2048 x 2048
single-precision, with `alpha = 32412` and `beta = 2123`, seeded by
`A[i][j] = B[i][j] = C[i][j] = (i*j) / NI`. Sizes are compile-time constants in
`gemm.cuh`.

## Project Structure

```
.
├── common/                                    # Vendored PolyBench 3.2 harness
│   ├── polybench.c                            # Timer + allocator implementation
│   ├── polybench.h                            # Array-decl and instrument macros
│   └── polybenchUtilFuncts.h                  # percentDiff() result checker
├── CUDA/
│   ├── common.mk                              # Shared nvcc rules (one per .cu)
│   └── GEMM/
│       ├── gemm.cuh                           # Dataset sizes, block dims, DATA_TYPE
│       ├── Makefile                           # Lists the three variants
│       ├── gemm_tiled_double_buffering.cu     # Variant 1
│       ├── gemm_tiled_warp_specialization.cu  # Variant 2
│       └── gemm_cooperative_pipeline.cu       # Variant 3
└── README.md
```

Each `.cu` is a standalone program: it carries its own `main`, its own CPU
reference, and `#include`s `common/polybench.c` directly at the bottom, so
there is exactly one translation unit per binary.

## Build Instructions

```bash
cd CUDA/GEMM
make
```

`common.mk` builds every file listed in `CUFILES` with identical flags
(`-O3 -std=c++17 -arch=sm_80`), so the three timings are comparable. This
produces `gemm_tiled_double_buffering.exe`, `gemm_tiled_warp_specialization.exe`
and `gemm_cooperative_pipeline.exe`.

To target a different GPU, override the arch without editing the makefile:

```bash
make NVCCFLAGS="-O3 -std=c++17 -arch=sm_90"
```

`make clean` removes the binaries. Note that `-std=c++17` and `-arch=sm_80` are
hard requirements for variant 3 — `cuda::pipeline` and `cp.async` need Ampere
or newer.

## Run Instructions

Each binary takes no arguments; the problem size is fixed at compile time in
`gemm.cuh`.

```bash
./gemm_tiled_double_buffering.exe
./gemm_tiled_warp_specialization.exe
./gemm_cooperative_pipeline.exe
```

By default every variant is built with `RUN_ON_CPU` defined, so each run also
computes the reference GEMM on the CPU and checks the GPU result against it.
That CPU pass is a naive triple loop over 2048^3 and takes about 27 seconds —
it dominates the wall-clock time of a run and is not part of the reported GPU
time. Comment out `#define RUN_ON_CPU` in a variant to skip verification and
dump the result matrix to stderr instead.

## Output Format

```
setting device 0 with name NVIDIA A100 80GB PCIe
GPU Time in seconds:
0.003167
CPU Time in seconds:
27.714455
Non-Matching CPU-GPU Outputs Beyond Error Threshold of 0.05 Percent: 0
```

The GPU time covers the kernel launch and `cudaDeviceSynchronize` only — the
`cudaMalloc`s and the host<->device copies are outside the timed region, so
what is being compared is kernel time, not end-to-end transfer time. The last
line counts output elements where the GPU and CPU differ by more than 0.05
percent; anything other than `0` means the variant is wrong.

## Notes

- All three variants are built on the same harness and are structurally
  identical outside the kernel: same `init`, same CPU reference `gemm`, same
  `compareResults`, same `gemmCuda` skeleton (allocate, copy in, time the
  launch, check for errors, copy out, free), and the same `main`. The only
  differences are the header comment, the variant's own tuning constants and
  includes, the grid/block setup, and the kernel call itself.
- Each kernel's name matches the name of the file it lives in.
- Variant 2 uses a 32 x 8 block so that a warp is exactly one `threadIdx.y`
  row and `threadIdx.x` is a lane id. `RESOURCE_PER_SM` is derived from the
  block shape, so the FP32/FP64 slot count follows the warp count if the block
  is retuned.
- Variant 3 operates on square matrices whose size is a multiple of `BUFF_DIM`
  (32), and uses a 32 x 32 block (1024 threads).

## System Configuration

**GPU:** NVIDIA A100 80GB PCIe (compute capability 8.0, 81920 MiB)

**CUDA:**
- CUDA Toolkit / nvcc: 13.0 (V13.0.88)
- Driver: 580.82.07

**OS:** Ubuntu 22.04.1 LTS (kernel 5.15.0-71-generic)

**Compiler flags:** `-O3 -std=c++17 -arch=sm_80`

**Dataset:** PolyBench/GPU standard GEMM, `NI = NJ = NK = 2048`,
single precision (`DATA_TYPE float`), `alpha = 32412`, `beta = 2123`

**Benchmark harness:** PolyBench/C 3.2 timing macros via PolyBench/GPU 1.0
(Scott Grauer-Gray, Will Killian, Louis-Noel Pouchet)
