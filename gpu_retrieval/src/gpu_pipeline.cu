#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "gpu_pipeline.h"

#define CK(call)                                                                              \
    do {                                                                                      \
        cudaError_t e_ = (call);                                                              \
        if (e_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

static const int TILE = 32;
static const int TPB = 256;   // threads per block for top-k kernels
static const int KMAX = 32;

// ---------------------------------------------------------------------------
// P1: L2 normalization. One block per vector, shared-memory tree reduction.
// ---------------------------------------------------------------------------
__global__ void normalize_rows(float* X, int D) {
    float* v = X + (size_t)blockIdx.x * D;
    __shared__ float sh[128];
    int t = threadIdx.x;
    float ss = 0.f;
    for (int d = t; d < D; d += blockDim.x) ss += v[d] * v[d];
    sh[t] = ss;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) sh[t] += sh[t + off];
        __syncthreads();
    }
    float inv = rsqrtf(sh[0] + 1e-12f);
    for (int d = t; d < D; d += blockDim.x) v[d] *= inv;
}

// ---------------------------------------------------------------------------
// P2: scores[q][n] = dot(Qm[q], C[n]). 32x32 tiles over (query, corpus), loop over D.
// Loads are coalesced along D; Bs is padded (+1) so Bs[tx][kk] is bank-conflict free.
// ---------------------------------------------------------------------------
__global__ void score_tiled(const float* __restrict__ Qm, const float* __restrict__ C, float* __restrict__ S,
                            int Q, int N, int D) {
    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int q = blockIdx.y * TILE + ty;
    int n = blockIdx.x * TILE + tx;
    int nrow = blockIdx.x * TILE + ty;  // corpus row this thread loads into shared memory
    float acc = 0.f;
    for (int d0 = 0; d0 < D; d0 += TILE) {
        int d = d0 + tx;
        As[ty][tx] = (q < Q && d < D) ? Qm[(size_t)q * D + d] : 0.f;
        Bs[ty][tx] = (nrow < N && d < D) ? C[(size_t)nrow * D + d] : 0.f;
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE; kk++) acc += As[ty][kk] * Bs[tx][kk];
        __syncthreads();
    }
    if (q < Q && n < N) S[(size_t)q * N + n] = acc;
}

// ---------------------------------------------------------------------------
// P3 (ours): block-level top-k. One block per segment.
// Each thread scans a strided slice keeping a private sorted top-k list, then the block
// merges the TPB lists with k rounds of block-wide argmax (each round pops one list head).
// Used twice: stage 1 over slices of each score row, stage 2 over the stage-1 candidates.
//
// Segment s -> query q = s / B, slice b = s % B. Slice covers [b*seg_len, +seg_len) of the row
// (clipped to row_len). If ids == nullptr the candidate id is its position in the row.
// ---------------------------------------------------------------------------
__global__ void topk_block(const float* __restrict__ vals, const int* __restrict__ ids, size_t row_stride,
                           int row_len, int B, int seg_len, int K, float* __restrict__ out_v,
                           int* __restrict__ out_i) {
    int s = blockIdx.x;
    int q = s / B, b = s % B;
    int start = b * seg_len;
    int len = min(seg_len, row_len - start);
    const float* v = vals + (size_t)q * row_stride + start;
    const int* id = ids ? ids + (size_t)q * row_stride + start : nullptr;

    float lv[KMAX];
    int li[KMAX];
    for (int j = 0; j < K; j++) { lv[j] = -INFINITY; li[j] = -1; }

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        float x = v[i];
        if (x > lv[K - 1]) {  // insertion into sorted (descending) list
            int gi = id ? id[i] : (start + i);
            int j = K - 1;
            while (j > 0 && lv[j - 1] < x) { lv[j] = lv[j - 1]; li[j] = li[j - 1]; j--; }
            lv[j] = x;
            li[j] = gi;
        }
    }

    __shared__ float sv[TPB];
    __shared__ int st[TPB];
    int t = threadIdx.x;
    int head = 0;
    for (int r = 0; r < K; r++) {
        sv[t] = (head < K) ? lv[head] : -INFINITY;
        st[t] = t;
        __syncthreads();
        for (int off = TPB / 2; off > 0; off >>= 1) {
            if (t < off && sv[t + off] > sv[t]) { sv[t] = sv[t + off]; st[t] = st[t + off]; }
            __syncthreads();
        }
        int w = st[0];
        if (t == w) {  // winner writes its head and advances
            out_v[(size_t)s * K + r] = lv[head];
            out_i[(size_t)s * K + r] = li[head];
            head++;
        }
        __syncthreads();  // everyone has read sv[0]/st[0] before the next round overwrites them
    }
}

// ---------------------------------------------------------------------------
// Host pipeline
// ---------------------------------------------------------------------------
template <typename F>
static void timed(float& acc, F f) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    acc += ms;
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
}

void gpu_print_device() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s, %d SMs, %.1f GB, cc %d.%d\n", p.name, p.multiProcessorCount, p.totalGlobalMem / 1e9,
           p.major, p.minor);
}

void gpu_search(const float* hostC, const float* hostQ, int Q, int N, int D, int K, int B, int qbatch,
                TopkMode mode, Result& out, GpuTimings& t) {
    t = GpuTimings();
    int dev = 0;
    CK(cudaGetDevice(&dev));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

    if (qbatch <= 0) qbatch = (int)std::max<long long>(1, std::min<long long>(Q, (1LL << 27) / N));
    qbatch = std::min(qbatch, Q);
    if (B <= 0) {
        B = (8 * sms + qbatch - 1) / qbatch;                  // enough blocks to fill the GPU
        B = std::min(B, std::max(1, N / (TPB * 4)));         // keep >= ~4 elements per thread
    }
    B = std::max(B, 1);
    int seg_len = (N + B - 1) / B;

    float *dC, *dQ, *dS, *dV1 = nullptr, *dOutS;
    int *dI1 = nullptr, *dOutI, *dIdx = nullptr;
    CK(cudaMalloc(&dC, (size_t)N * D * sizeof(float)));
    CK(cudaMalloc(&dQ, (size_t)Q * D * sizeof(float)));
    CK(cudaMalloc(&dS, (size_t)qbatch * N * sizeof(float)));
    CK(cudaMalloc(&dOutS, (size_t)Q * K * sizeof(float)));
    CK(cudaMalloc(&dOutI, (size_t)Q * K * sizeof(int)));
    if (mode == TOPK_CUSTOM) {
        CK(cudaMalloc(&dV1, (size_t)qbatch * B * K * sizeof(float)));
        CK(cudaMalloc(&dI1, (size_t)qbatch * B * K * sizeof(int)));
    } else {
        CK(cudaMalloc(&dIdx, (size_t)N * sizeof(int)));
    }

    // H2D
    timed(t.h2d, [&] {
        CK(cudaMemcpy(dC, hostC, (size_t)N * D * sizeof(float), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dQ, hostQ, (size_t)Q * D * sizeof(float), cudaMemcpyHostToDevice));
    });

    // P1 preprocess
    timed(t.preprocess, [&] {
        normalize_rows<<<N, 128>>>(dC, D);
        normalize_rows<<<Q, 128>>>(dQ, D);
        CK(cudaGetLastError());
    });

    // P2 + P3 per query batch
    for (int q0 = 0; q0 < Q; q0 += qbatch) {
        int qb = std::min(qbatch, Q - q0);

        timed(t.score, [&] {
            dim3 block(TILE, TILE);
            dim3 grid((N + TILE - 1) / TILE, (qb + TILE - 1) / TILE);
            score_tiled<<<grid, block>>>(dQ + (size_t)q0 * D, dC, dS, qb, N, D);
            CK(cudaGetLastError());
        });

        timed(t.topk, [&] {
            if (mode == TOPK_CUSTOM) {
                topk_block<<<qb * B, TPB>>>(dS, nullptr, (size_t)N, N, B, seg_len, K, dV1, dI1);
                CK(cudaGetLastError());
                int cand = B * K;
                topk_block<<<qb, TPB>>>(dV1, dI1, (size_t)cand, cand, 1, cand, K,
                                        dOutS + (size_t)q0 * K, dOutI + (size_t)q0 * K);
                CK(cudaGetLastError());
            } else {
                for (int q = 0; q < qb; q++) {
                    thrust::device_ptr<float> s(dS + (size_t)q * N);
                    thrust::device_ptr<int> idx(dIdx);
                    thrust::sequence(thrust::device, idx, idx + N);
                    thrust::sort_by_key(thrust::device, s, s + N, idx, thrust::greater<float>());
                    CK(cudaMemcpy(dOutS + (size_t)(q0 + q) * K, dS + (size_t)q * N, K * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
                    CK(cudaMemcpy(dOutI + (size_t)(q0 + q) * K, dIdx, K * sizeof(int),
                                  cudaMemcpyDeviceToDevice));
                }
            }
        });
    }

    // D2H
    out.s.resize((size_t)Q * K);
    out.id.resize((size_t)Q * K);
    timed(t.d2h, [&] {
        CK(cudaMemcpy(out.s.data(), dOutS, (size_t)Q * K * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out.id.data(), dOutI, (size_t)Q * K * sizeof(int), cudaMemcpyDeviceToHost));
    });

    t.total = t.h2d + t.preprocess + t.score + t.topk + t.d2h;

    CK(cudaFree(dC));
    CK(cudaFree(dQ));
    CK(cudaFree(dS));
    CK(cudaFree(dOutS));
    CK(cudaFree(dOutI));
    if (dV1) CK(cudaFree(dV1));
    if (dI1) CK(cudaFree(dI1));
    if (dIdx) CK(cudaFree(dIdx));
}
