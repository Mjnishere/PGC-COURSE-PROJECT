#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

struct Config {
    int N = 262144;   // corpus size
    int D = 128;      // embedding dimension
    int Q = 64;       // number of queries
    int K = 10;       // top-k (<= 32)
    int reps = 3;     // timing repetitions (best-of)
    int B = 0;        // GPU blocks per query for stage 1 (0 = auto)
    int qbatch = 0;   // GPU query batch size (0 = auto)
    unsigned seed = 42;
    bool csv = false;
};

inline Config parse_args(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? atoi(argv[++i]) : 0; };
        if (a == "--N") c.N = next();
        else if (a == "--D") c.D = next();
        else if (a == "--Q") c.Q = next();
        else if (a == "--K") c.K = next();
        else if (a == "--reps") c.reps = next();
        else if (a == "--B") c.B = next();
        else if (a == "--qbatch") c.qbatch = next();
        else if (a == "--seed") c.seed = (unsigned)next();
        else if (a == "--csv") c.csv = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); exit(1); }
    }
    if (c.K < 1 || c.K > 32) { fprintf(stderr, "K must be in [1,32]\n"); exit(1); }
    return c;
}

// Deterministic raw data: uniform [-1,1] times a per-row random scale in [0.5,5],
// so that the normalization step really matters.
inline void gen_data(std::vector<float>& X, int rows, int D, unsigned seed) {
    X.resize((size_t)rows * D);
    uint32_t s = seed * 2654435761u + 12345u;
    auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >> 8) * (1.0f / 16777216.0f); };
    for (int r = 0; r < rows; r++) {
        float scale = 0.5f + 4.5f * rnd();
        for (int d = 0; d < D; d++) X[(size_t)r * D + d] = (2.0f * rnd() - 1.0f) * scale;
    }
}

struct Timer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

// Q x K results: scores sorted descending, ids of the matching corpus vectors.
struct Result {
    std::vector<float> s;
    std::vector<int> id;
};

// Compare against a reference. Returns recall@K (fraction of ids also in the reference)
// and the maximum absolute score difference at equal rank.
inline void verify(const Result& ref, const Result& got, int Q, int K, double& recall, double& maxdiff) {
    size_t hit = 0;
    maxdiff = 0;
    for (int q = 0; q < Q; q++) {
        std::unordered_set<int> r(ref.id.begin() + (size_t)q * K, ref.id.begin() + (size_t)(q + 1) * K);
        for (int j = 0; j < K; j++) {
            hit += r.count(got.id[(size_t)q * K + j]);
            maxdiff = std::max(maxdiff, (double)std::fabs(ref.s[(size_t)q * K + j] - got.s[(size_t)q * K + j]));
        }
    }
    recall = (double)hit / ((double)Q * K);
}
