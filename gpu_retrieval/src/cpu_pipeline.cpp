#include "cpu_pipeline.h"
#include <omp.h>

void cpu_normalize(float* X, int rows, int D, int threads) {
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int r = 0; r < rows; r++) {
        float* v = X + (size_t)r * D;
        float ss = 0.f;
        for (int d = 0; d < D; d++) ss += v[d] * v[d];
        float inv = 1.0f / std::sqrt(ss + 1e-12f);
        for (int d = 0; d < D; d++) v[d] *= inv;
    }
}

void cpu_search(const float* Qm, const float* C, int Q, int N, int D, int K, int threads, Result& out) {
    out.s.assign((size_t)Q * K, -INFINITY);
    out.id.assign((size_t)Q * K, -1);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1)
    for (int q = 0; q < Q; q++) {
        const float* qv = Qm + (size_t)q * D;
        float bs[32];
        int bi[32];
        for (int j = 0; j < K; j++) { bs[j] = -INFINITY; bi[j] = -1; }
        for (int n = 0; n < N; n++) {
            const float* cv = C + (size_t)n * D;
            float dot = 0.f;
            for (int d = 0; d < D; d++) dot += qv[d] * cv[d];
            if (dot > bs[K - 1]) {  // insertion into the sorted top-K list
                int j = K - 1;
                while (j > 0 && bs[j - 1] < dot) { bs[j] = bs[j - 1]; bi[j] = bi[j - 1]; j--; }
                bs[j] = dot;
                bi[j] = n;
            }
        }
        for (int j = 0; j < K; j++) {
            out.s[(size_t)q * K + j] = bs[j];
            out.id[(size_t)q * K + j] = bi[j];
        }
    }
}
