/**
 * gemm_tiled_double_buffering.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * GEMM using shared-memory tiling with double-buffered tile loading: half of
 * each block's threads prefetch the next pair of A/B tiles while the other
 * half multiply the current pair.
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

__global__ void gemm_tiled_double_buffering(int ni, int nj, int nk,
                                            const DATA_TYPE alpha, const DATA_TYPE beta,
                                            const DATA_TYPE *__restrict__ a,
                                            const DATA_TYPE *__restrict__ b,
                                            DATA_TYPE *__restrict__ c)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int TILE_SIZE = DIM_THREAD_BLOCK_X;

    __shared__ DATA_TYPE Asub1[TILE_SIZE][TILE_SIZE];
    __shared__ DATA_TYPE Bsub1[TILE_SIZE][TILE_SIZE];
    __shared__ DATA_TYPE Asub2[TILE_SIZE][TILE_SIZE];
    __shared__ DATA_TYPE Bsub2[TILE_SIZE][TILE_SIZE];
    DATA_TYPE val1 = 0;
    DATA_TYPE val2 = 0;

    DATA_TYPE *Asubptr = &Asub1[0][0];
    DATA_TYPE *Bsubptr = &Bsub1[0][0];

    if (threadIdx.y < TILE_SIZE / 2)
    {
        int isub1 = 2 * threadIdx.y;
        int isub2 = isub1 + 1;
        int i1 = blockIdx.y * blockDim.y + isub1;
        int i2 = blockIdx.y * blockDim.y + isub2;
        if (i1 < ni && threadIdx.x < nk)
        {
            Asubptr[isub1 * TILE_SIZE + threadIdx.x] = a[i1 * nk + (threadIdx.x)];
            Asubptr[isub2 * TILE_SIZE + threadIdx.x] = a[i2 * nk + (threadIdx.x)];
        }
        else
        {
            Asubptr[isub1 * TILE_SIZE + threadIdx.x] = 0.0;
            Asubptr[isub2 * TILE_SIZE + threadIdx.x] = 0.0;
        }
        if (j < nj && isub1 < nk)
        {
            Bsubptr[isub1 * TILE_SIZE + threadIdx.x] = b[(isub1)*nj + j];
            Bsubptr[isub2 * TILE_SIZE + threadIdx.x] = b[(isub2)*nj + j];
        }
        else
        {
            Bsubptr[isub1 * TILE_SIZE + threadIdx.x] = 0.0;
            Bsubptr[isub2 * TILE_SIZE + threadIdx.x] = 0.0;
        }
    }
    DATA_TYPE *Asubptr2 = Asubptr;
    DATA_TYPE *Bsubptr2 = Bsubptr;
    Asubptr = &Asub2[0][0];
    Bsubptr = &Bsub2[0][0];

    __syncthreads();

    for (int t = 0; t < ((nk + TILE_SIZE - 1) / TILE_SIZE) - 1; t++)
    {
        if (threadIdx.y < TILE_SIZE / 2)
        {
            int isub1 = 2 * threadIdx.y;
            int isub2 = isub1 + 1;
            int i1 = blockIdx.y * blockDim.y + isub1;
            int i2 = blockIdx.y * blockDim.y + isub2;
            if (i1 < ni && ((t + 1) * TILE_SIZE + threadIdx.x) < nk)
            {
                Asubptr[isub1 * TILE_SIZE + threadIdx.x] = a[i1 * nk + ((t + 1) * TILE_SIZE + threadIdx.x)];
                Asubptr[isub2 * TILE_SIZE + threadIdx.x] = a[i2 * nk + ((t + 1) * TILE_SIZE + threadIdx.x)];
            }
            else
            {
                Asubptr[isub1 * TILE_SIZE + threadIdx.x] = 0.0;
                Asubptr[isub2 * TILE_SIZE + threadIdx.x] = 0.0;
            }
            if (j < nj && ((t + 1) * TILE_SIZE + isub1) < nk)
            {
                Bsubptr[isub1 * TILE_SIZE + threadIdx.x] = b[((t + 1) * TILE_SIZE + isub1) * nj + j];
                Bsubptr[isub2 * TILE_SIZE + threadIdx.x] = b[((t + 1) * TILE_SIZE + isub2) * nj + j];
            }
            else
            {
                Bsubptr[isub1 * TILE_SIZE + threadIdx.x] = 0.0;
                Bsubptr[isub2 * TILE_SIZE + threadIdx.x] = 0.0;
            }
        }
        else
        {
            for (int k = 0; k < TILE_SIZE; k++)
            {
                val1 += Asubptr2[2 * (threadIdx.y - TILE_SIZE / 2) * TILE_SIZE + k] * Bsubptr2[k * TILE_SIZE + threadIdx.x];
                val2 += Asubptr2[(2 * (threadIdx.y - TILE_SIZE / 2) + 1) * TILE_SIZE + k] * Bsubptr2[k * TILE_SIZE + threadIdx.x];
            }
        }

        DATA_TYPE *Asubtmp = Asubptr;
        DATA_TYPE *Bsubtmp = Bsubptr;
        Asubptr = Asubptr2;
        Bsubptr = Bsubptr2;
        Asubptr2 = Asubtmp;
        Bsubptr2 = Bsubtmp;

        __syncthreads();
    }
    if (threadIdx.y >= TILE_SIZE / 2)
    {
        for (int k = 0; k < TILE_SIZE; k++)
        {
            val1 += Asubptr2[2 * (threadIdx.y - TILE_SIZE / 2) * TILE_SIZE + k] * Bsubptr2[k * TILE_SIZE + threadIdx.x];
            val2 += Asubptr2[(2 * (threadIdx.y - TILE_SIZE / 2) + 1) * TILE_SIZE + k] * Bsubptr2[k * TILE_SIZE + threadIdx.x];
        }
        int isub1 = 2 * (threadIdx.y - TILE_SIZE / 2);
        int isub2 = isub1 + 1;
        int i1 = blockIdx.y * blockDim.y + isub1;
        int i2 = blockIdx.y * blockDim.y + isub2;
        if ((i1 < _PB_NI) && (j < _PB_NJ))
        {
            c[i1 * NJ + j] = c[i1 * NJ + j] * beta + val1 * alpha;
            c[i2 * NJ + j] = c[i2 * NJ + j] * beta + val2 * alpha;
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

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((size_t)(ceil(((float)NI) / ((float)block.x))), (size_t)(ceil(((float)NJ) / ((float)block.y))));

    /* Start timer. */
    polybench_start_instruments;

    gemm_tiled_double_buffering<<<grid, block>>>(ni, nj, nk, alpha, beta, A_gpu, B_gpu, C_gpu);
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
