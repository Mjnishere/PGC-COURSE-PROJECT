#pragma once
#include "common.h"

// P1: L2-normalize each of `rows` vectors of dimension D in place.
void cpu_normalize(float* X, int rows, int D, int threads);

// P2+P3 fused: for each query, scan the corpus and keep the K best dot products.
// Queries and corpus must already be normalized. Parallel over queries (OpenMP).
void cpu_search(const float* Qm, const float* C, int Q, int N, int D, int K, int threads, Result& out);
