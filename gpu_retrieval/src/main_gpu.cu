#include <omp.h>
#include <cuda_runtime.h>

#include "cpu_pipeline.h"
#include "gpu_pipeline.h"

static void run_gpu(const Config& c, const std::vector<float>& C, const std::vector<float>& Q, TopkMode m,
                    Result& res, GpuTimings& best) {
    best.total = 1e30f;
    gpu_search(C.data(), Q.data(), c.Q, c.N, c.D, c.K, c.B, c.qbatch, m, res, best);  // warm-up
    best.total = 1e30f;
    for (int r = 0; r < c.reps; r++) {
        GpuTimings t;
        gpu_search(C.data(), Q.data(), c.Q, c.N, c.D, c.K, c.B, c.qbatch, m, res, t);
        if (t.total < best.total) best = t;
    }
}

int main(int argc, char** argv) {
    Config c = parse_args(argc, argv);
    cudaFree(0);  // create context outside any timing
    if (!c.csv) gpu_print_device();

    std::vector<float> rawC, rawQ;
    gen_data(rawC, c.N, c.D, c.seed);
    gen_data(rawQ, c.Q, c.D, c.seed + 1);
    int nt = omp_get_max_threads();

    // CPU reference (OpenMP), best of reps
    Result ref;
    double cpre = 1e30, csearch = 1e30;
    for (int r = 0; r < c.reps; r++) {
        std::vector<float> C = rawC, Q = rawQ;
        Timer t;
        t.start();
        cpu_normalize(C.data(), c.N, c.D, nt);
        cpu_normalize(Q.data(), c.Q, c.D, nt);
        double a = t.ms();
        t.start();
        cpu_search(Q.data(), C.data(), c.Q, c.N, c.D, c.K, nt, ref);
        double b = t.ms();
        cpre =std::min(cpre, a);
        csearch = std::min(csearch, b);
    }
    double ctotal = cpre + csearch;

    Result r_custom, r_thrust;
    GpuTimings t_custom, t_thrust;
    run_gpu(c, rawC, rawQ, TOPK_CUSTOM, r_custom, t_custom);
    run_gpu(c, rawC, rawQ, TOPK_THRUST, r_thrust, t_thrust);

    double rec1, md1, rec2, md2;
    verify(ref, r_custom, c.Q, c.K, rec1, md1);
    verify(ref, r_thrust, c.Q, c.K, rec2, md2);

    if (c.csv) {
        printf("impl,N,D,Q,K,h2d_ms,preprocess_ms,score_ms,topk_ms,d2h_ms,total_ms,recall,maxdiff,B\n");
        printf("cpu_omp,%d,%d,%d,%d,0,%.3f,%.3f,0,0,%.3f,1,0,%d\n", c.N, c.D, c.Q, c.K, cpre, csearch, ctotal, c.B);
        auto row = [&](const char* n, const GpuTimings& t, double rec, double md) {
            printf("%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.2e,%d\n", n, c.N, c.D, c.Q, c.K, t.h2d,
                   t.preprocess, t.score, t.topk, t.d2h, t.total, rec, md, c.B);
        };
        row("gpu_custom_topk", t_custom, rec1, md1);
        row("gpu_thrust_sort", t_thrust, rec2, md2);
    } else {
        printf("N=%d D=%d Q=%d K=%d  (CPU reference: OpenMP, %d threads)\n", c.N, c.D, c.Q, c.K, nt);
        printf("%-16s %8s %8s %8s %8s %8s %9s %8s\n", "impl", "h2d", "preproc", "score", "topk", "d2h", "total",
               "speedup");
        printf("%-16s %8s %8.2f %8.2f %8s %8s %9.2f %8s\n", "cpu_omp", "-", cpre, csearch, "(fused)", "-", ctotal,
               "1.00x");
        auto row = [&](const char* n, const GpuTimings& t) {
            printf("%-16s %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f %7.1fx\n", n, t.h2d, t.preprocess, t.score, t.topk,
                   t.d2h, t.total, ctotal / t.total);
        };
        row("gpu_custom_topk", t_custom);
        row("gpu_thrust_sort", t_thrust);
        printf("correctness vs CPU: custom recall=%.4f maxdiff=%.1e | thrust recall=%.4f maxdiff=%.1e\n", rec1, md1,
               rec2, md2);
        printf("(all times in ms)\n");
    }
    return 0;
}
