/**
 * gemm_cooperative_pipeline.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * GEMM using cooperative groups and cuda::pipeline async copies, overlapping
 * global->shared tile fetches with compute across a two-stage buffer.
 * Ported from Maharshi's standalone matmul_2s.cu.
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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

#define GPU_DEVICE 0

// define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412.0f
#define BETA 2123.0f

#define RUN_ON_CPU

#define THREADS_PER_BLOCK 32
#define BUFF_DIM 32
#define NUM_STAGES 2

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

__global__ void gemm_cooperative_pipeline(int n, DATA_TYPE alpha, DATA_TYPE beta,
                                          DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
    __shared__ DATA_TYPE buff_B[NUM_STAGES][BUFF_DIM][BUFF_DIM];
    __shared__ DATA_TYPE buff_A[NUM_STAGES][BUFF_DIM][BUFF_DIM];

    auto block = cooperative_groups::this_thread_block();
    auto thread = cooperative_groups::this_thread();
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
    int n_iters = n / BUFF_DIM;
    int x = blockIdx.x * BUFF_DIM;
    int y = blockIdx.y * BUFF_DIM;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    DATA_TYPE res = 0.0;

    for (size_t compute_batch = 0, fetch_batch = 0; compute_batch < n_iters; ++compute_batch)
    {
        for (; fetch_batch < n_iters && fetch_batch < (compute_batch + NUM_STAGES); ++fetch_batch)
        {
            pipeline.producer_acquire();
            size_t shared_idx = fetch_batch % NUM_STAGES;
            size_t batch_idx = fetch_batch;

            if (tx * 4 < BUFF_DIM)
            {
                cuda::memcpy_async(thread, &buff_A[shared_idx][ty][tx * 4], &A[x * n + batch_idx * BUFF_DIM + ty * n + tx * 4], 4 * sizeof(DATA_TYPE), pipeline);
                cuda::memcpy_async(thread, &buff_B[shared_idx][ty][tx * 4], &B[y + batch_idx * BUFF_DIM * n + ty * n + tx * 4], 4 * sizeof(DATA_TYPE), pipeline);
            }

            pipeline.producer_commit();
        }

        pipeline.consumer_wait();
        block.sync();
        int shared_idx = compute_batch % NUM_STAGES;

        for (int j = 0; j < BUFF_DIM; j++)
        {
            res += buff_A[shared_idx][ty][j] * buff_B[shared_idx][j][tx];
        }

        pipeline.consumer_release();
        block.sync();
    }

    int idx = x * n + y + ty * n + tx;
    C[idx] = beta * C[idx] + alpha * res;
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

    int n = ni;
    int blocks = n / BUFF_DIM;

    dim3 block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid(blocks, blocks);

    /* Start timer. */
    polybench_start_instruments;

    gemm_cooperative_pipeline<<<grid, block>>>(n, alpha, beta, A_gpu, B_gpu, C_gpu);
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
