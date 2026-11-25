#pragma once
#include "common.h"

enum TopkMode { TOPK_CUSTOM = 0, TOPK_THRUST = 1 };

struct GpuTimings {
    float h2d = 0, preprocess = 0, score = 0, topk = 0, d2h = 0, total = 0;  // milliseconds
};

// Full GPU pipeline: H2D copy, normalize (P1), tiled scoring (P2), top-k (P3), D2H copy.
// hostC: N x D raw corpus, hostQ: Q x D raw queries (both un-normalized).
// B = stage-1 blocks per query (0 = auto), qbatch = queries per batch (0 = auto).
void gpu_search(const float* hostC, const float* hostQ, int Q, int N, int D, int K, int B, int qbatch,
                TopkMode mode, Result& out, GpuTimings& t);

void gpu_print_device();
