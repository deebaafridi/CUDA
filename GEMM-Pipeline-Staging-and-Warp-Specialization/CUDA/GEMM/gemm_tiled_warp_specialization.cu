/**
 * gemm_tiled_warp_specialization.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * GEMM in which each warp reserves either an FP32 or an FP64 execution
 * resource for the duration of its dot product, spreading a block's warps
 * across both unit types.
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 *          Will Killian <killian@udel.edu>
 *          Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gemm.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

#define WARP_SIZE 32
#define DIM_THREAD_BLOCK_WARP_X WARP_SIZE
#define DIM_THREAD_BLOCK_WARP_Y 8
#define WARPS_PER_BLOCK ((DIM_THREAD_BLOCK_WARP_X * DIM_THREAD_BLOCK_WARP_Y) / WARP_SIZE)
#define RESOURCE_PER_SM (WARPS_PER_BLOCK / 2)

void init(int ni, int nj, int nk, DATA_TYPE *alpha, DATA_TYPE *beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;

    *alpha = ALPHA;
    *beta = BETA;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nk; j++)
        {
            A[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < nk; i++)
    {
        for (j = 0; j < nj; j++)
        {
            B[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i][j] = ((DATA_TYPE)i * j) / NI;
        }
    }
}

void gemm(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
          DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
          DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
          DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j, k;

    for (i = 0; i < _PB_NI; i++)
    {
        for (j = 0; j < _PB_NJ; j++)
        {
            C[i][j] *= beta;

            for (k = 0; k < _PB_NK; ++k)
            {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

void compareResults(int ni, int nj,
                    DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                    DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj))
{
    int i, j, fail;
    fail = 0;

    // Compare CPU and GPU outputs
    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            if (percentDiff(C[i][j], C_outputFromGpu[i][j]) > PERCENT_DIFF_ERROR_THRESHOLD)
            {
                fail++;
            }
        }
    }

    printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}

void GPU_argv_init()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
    printf("setting device %d with name %s\n", GPU_DEVICE, deviceProp.name);
    cudaSetDevice(GPU_DEVICE);
}

__global__ void gemm_tiled_warp_specialization(int ni, int nj, int nk,
                                               DATA_TYPE alpha, DATA_TYPE beta,
                                               DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    __shared__ int fp32CountInSM;
    __shared__ int fp64CountInSM;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        fp32CountInSM = 0;
        fp64CountInSM = 0;
    }
    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= ni || col >= nj)
        return;

    unsigned threadIdInWarp = threadIdx.x % WARP_SIZE;
    bool usingFP32Unit = false;

    if (threadIdInWarp == 0)
    {
        bool occupyResourceUnit = false;
        while (!occupyResourceUnit)
        {
            int currentStatusOfFP32 = atomicAdd(&fp32CountInSM, 0);
            if (currentStatusOfFP32 < RESOURCE_PER_SM)
            {
                if (atomicCAS(&fp32CountInSM, currentStatusOfFP32, currentStatusOfFP32 + 1) == currentStatusOfFP32)
                {
                    usingFP32Unit = true;
                    occupyResourceUnit = true;
                    break;
                }
            }
            int currentStatusOfFP64 = atomicAdd(&fp64CountInSM, 0);
            if (currentStatusOfFP64 < RESOURCE_PER_SM)
            {
                if (atomicCAS(&fp64CountInSM, currentStatusOfFP64, currentStatusOfFP64 + 1) == currentStatusOfFP64)
                {
                    usingFP32Unit = false;
                    occupyResourceUnit = true;
                    break;
                }
            }
        }
    }
    __syncwarp();

    bool warpFP32Choice = __shfl_sync(0xFFFFFFFF, usingFP32Unit, 0);

    double product_d = 0.0;
    float product_f = 0.0f;
    int idx = row * nj + col;

    if (warpFP32Choice)
    {
        for (int k = 0; k < nk; ++k)
        {
            float a = A[row * nk + k];
            float b = B[k * nj + col];
            product_f += alpha * a * b;
        }
        C[idx] = beta * C[idx] + product_f;
    }
    else
    {
        for (int k = 0; k < nk; ++k)
        {
            float a = A[row * nk + k];
            float b = B[k * nj + col];
            product_d += static_cast<double>(alpha) * a * b;
        }
        C[idx] = beta * C[idx] + static_cast<float>(product_d);
    }
    __syncwarp();

    if (threadIdInWarp == 0)
    {
        if (warpFP32Choice)
        {
            atomicSub(&fp32CountInSM, 1);
        }
        else
        {
            atomicSub(&fp64CountInSM, 1);
        }
    }
}

void gemmCuda(int ni, int nj, int nk, DATA_TYPE alpha, DATA_TYPE beta,
              DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
              DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
              DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
              DATA_TYPE POLYBENCH_2D(C_outputFromGpu, NI, NJ, ni, nj))
{
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NI * NJ);

    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_WARP_X, DIM_THREAD_BLOCK_WARP_Y);
    dim3 grid((nj + block.x - 1) / block.x, (ni + block.y - 1) / block.y);

    /* Start timer. */
    polybench_start_instruments;

    gemm_tiled_warp_specialization<<<grid, block>>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
    cudaDeviceSynchronize();

    /* Stop and print timer. */
    polybench_stop_instruments;
    printf("GPU Time in seconds:\n");
    polybench_print_instruments;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error Name: %s\n", cudaGetErrorName(err));
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost);

    cudaFree(A_gpu);
    cudaFree(B_gpu);
    cudaFree(C_gpu);
}

#ifndef RUN_ON_CPU
/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
    int i, j;

    for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++)
        {
            fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
            if ((i * ni + j) % 20 == 0)
                fprintf(stderr, "\n");
        }
    fprintf(stderr, "\n");
}
#endif //! RUN_ON_CPU

int main(int argc, char *argv[])
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    int nk = NK;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
    POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_outputFromGpu, DATA_TYPE, NI, NJ, ni, nj);

    init(ni, nj, nk, &alpha, &beta,
         POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    GPU_argv_init();

    gemmCuda(ni, nj, nk, alpha, beta,
             POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#ifdef RUN_ON_CPU

    /* Start timer. */
    polybench_start_instruments;

    gemm(ni, nj, nk, alpha, beta,
         POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(C));

    /* Stop and print timer. */
    polybench_stop_instruments;
    printf("CPU Time in seconds:\n");
    polybench_print_instruments;

    compareResults(ni, nj, POLYBENCH_ARRAY(C), POLYBENCH_ARRAY(C_outputFromGpu));

#else // print output to stderr so no dead code elimination

    print_array(ni, nj, POLYBENCH_ARRAY(C_outputFromGpu));

#endif // RUN_ON_CPU

    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
    POLYBENCH_FREE_ARRAY(C);
    POLYBENCH_FREE_ARRAY(C_outputFromGpu);

    return 0;
}

#include "../../common/polybench.c"
